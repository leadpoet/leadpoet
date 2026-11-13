"""Local safety checks for supplemental backups; no AWS access or deployment."""

from pathlib import Path
import unittest

import yaml


TEMPLATE = Path(__file__).resolve().parents[1] / "deploy/aws/arena-recovery-backups.yml"
DATABASE = "arn:aws:rds:us-east-1:493765492819:db:leadpoet-subnet-arena"
VOLUMES = {
    "arn:aws:ec2:us-east-1:493765492819:volume/vol-0606194d5d5ed165d",
    "arn:aws:ec2:us-east-1:493765492819:volume/vol-0062edbcc50396de8",
    "arn:aws:ec2:us-east-1:493765492819:volume/vol-0bf023922624c5f40",
}
KEYS = {
    "arn:aws:kms:us-east-1:493765492819:key/89c50b1b-cb4e-463b-a4f1-b6025276f332",
    "arn:aws:kms:us-east-1:493765492819:key/5cba544d-bb01-4dca-849a-559ca365434d",
}


class CloudFormationLoader(yaml.SafeLoader):
    """Retain intrinsic functions as data; never evaluate CloudFormation code."""


def intrinsic(loader, tag, node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node)
    else:
        value = loader.construct_mapping(node)
    return {"Ref" if tag == "Ref" else "Fn::" + tag: value}


CloudFormationLoader.add_multi_constructor("!", intrinsic)


def items(value):
    return value if isinstance(value, list) else [value]


class ArenaRecoveryBackupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = yaml.load(TEMPLATE.read_text(), Loader=CloudFormationLoader)
        cls.resources = cls.template["Resources"]
        cls.role = cls.resources["BackupRole"]["Properties"]
        cls.statements = cls.role["Policies"][0]["PolicyDocument"]["Statement"]

    def statements_for(self, action):
        return [s for s in self.statements if action in items(s["Action"])]

    def test_stack_can_only_create_supplemental_backup_resources(self):
        types = [r["Type"] for r in self.resources.values()]
        self.assertCountEqual(types, ["AWS::Backup::BackupVault", "AWS::IAM::Role",
                                     "AWS::Backup::BackupPlan", "AWS::Backup::BackupPlan",
                                     "AWS::Backup::BackupSelection", "AWS::Backup::BackupSelection"])
        # No custom resource can run source SQL, change hosts or alter native PITR.
        self.assertNotIn("Transform", self.template)
        self.assertNotIn("ManagedPolicyArns", self.role)

    def test_each_schedule_applies_only_to_its_exact_source_selection(self):
        expected = {
            "DatabaseBackupSelection": ("DatabaseBackupPlan", {DATABASE}, "cron(0 11 * * ? *)", 35),
            "VolumeBackupSelection": ("VolumeBackupPlan", VOLUMES, "cron(0 12 * * ? *)", 14),
        }
        for selection_name, (plan_name, sources, schedule, days) in expected.items():
            with self.subTest(selection=selection_name):
                properties = self.resources[selection_name]["Properties"]
                self.assertEqual(properties["BackupPlanId"], {"Ref": plan_name})
                selection = properties["BackupSelection"]
                self.assertEqual(set(selection["Resources"]), sources)
                self.assertEqual(set(selection), {"SelectionName", "IamRoleArn", "Resources"})
                self.assertEqual(selection["IamRoleArn"], {"Fn::GetAtt": "BackupRole.Arn"})
                rules = self.resources[plan_name]["Properties"]["BackupPlan"]["BackupPlanRule"]
                self.assertEqual(len(rules), 1)
                rule = rules[0]
                self.assertEqual(rule["ScheduleExpression"], schedule)
                self.assertGreaterEqual(rule["StartWindowMinutes"], 60)
                self.assertFalse(rule["EnableContinuousBackup"])
                self.assertEqual(rule["Lifecycle"], {"DeleteAfterDays": days})
                self.assertEqual(rule["TargetBackupVault"], {"Ref": "RecoveryVault"})
                self.assertEqual(rule["RecoveryPointTags"]["leadpoet:backup-scope"], "arena-recovery")
                self.assertNotIn("CopyActions", rule)

    def test_vault_is_retained_with_optional_finite_compliance_lock(self):
        vault = self.resources["RecoveryVault"]
        self.assertEqual(vault["DeletionPolicy"], "Retain")
        self.assertEqual(vault["UpdateReplacePolicy"], "Retain")
        self.assertNotIn("EncryptionKeyArn", vault["Properties"])
        self.assertEqual(self.template["Parameters"]["EnableComplianceVaultLock"]["Default"], "true")
        self.assertEqual(vault["Properties"]["LockConfiguration"], {"Fn::If": [
            "UseComplianceVaultLock", {"ChangeableForDays": 7, "MinRetentionDays": 7,
                                       "MaxRetentionDays": 90}, {"Ref": "AWS::NoValue"}]})
        policy = vault["Properties"]["AccessPolicy"]["Statement"]
        self.assertTrue(all(s["Effect"] == "Deny" for s in policy))
        self.assertEqual(policy[0]["Condition"]["StringNotEquals"]["aws:PrincipalAccount"], "493765492819")

    def test_only_backup_service_in_this_account_and_region_can_assume_role(self):
        statements = self.role["AssumeRolePolicyDocument"]["Statement"]
        self.assertEqual(len(statements), 1)
        trust = statements[0]
        self.assertEqual(trust["Principal"], {"Service": "backup.amazonaws.com"})
        self.assertEqual(trust["Action"], "sts:AssumeRole")
        self.assertEqual(trust["Condition"], {
            "StringEquals": {"aws:SourceAccount": "493765492819"},
            "ArnLike": {"aws:SourceArn": "arn:aws:backup:us-east-1:493765492819:*"},
        })

    def test_snapshot_create_cannot_select_other_databases_or_volumes(self):
        rds = self.statements_for("rds:CreateDBSnapshot")
        self.assertEqual(len(rds), 1)
        self.assertEqual(set(rds[0]["Resource"]), {
            DATABASE, "arn:aws:rds:us-east-1:493765492819:snapshot:awsbackup:*"})
        ec2 = self.statements_for("ec2:CreateSnapshot")
        resources = {r for s in ec2 for r in items(s["Resource"])}
        self.assertEqual(resources, VOLUMES | {"arn:aws:ec2:us-east-1::snapshot/*"})

    def test_role_has_no_restore_source_delete_or_source_configuration_authority(self):
        actions = {a for s in self.statements for a in items(s["Action"])}
        self.assertTrue(all("*" not in a for a in actions))
        forbidden = {"rds:ModifyDBInstance", "rds:DeleteDBInstance",
                     "rds:DeleteDBInstanceAutomatedBackup", "rds:RestoreDBInstanceFromDBSnapshot",
                     "ec2:DeleteVolume", "ec2:CreateVolume", "ec2:RunInstances", "ec2:StopInstances",
                     "iam:PassRole", "ssm:SendCommand", "backup:StartRestoreJob"}
        self.assertFalse(actions & forbidden)
        for statement in self.statements:
            if statement["Resource"] == "*":
                self.assertTrue(all(":Describe" in a for a in items(statement["Action"])))
                self.assertEqual(statement["Condition"]["StringEquals"]["aws:RequestedRegion"], "us-east-1")

    def test_expiration_only_deletes_own_tagged_backup_snapshots(self):
        for action in ("rds:DeleteDBSnapshot", "ec2:DeleteSnapshot"):
            statements = self.statements_for(action)
            self.assertEqual(len(statements), 1)
            statement = statements[0]
            self.assertIn("snapshot", statement["Resource"])
            self.assertEqual(statement["Condition"]["StringEquals"]
                             ["aws:ResourceTag/leadpoet:backup-scope"], "arena-recovery")
        ec2 = self.statements_for("ec2:DeleteSnapshot")[0]
        self.assertEqual(ec2["Condition"]["Null"], {"aws:ResourceTag/aws:backup:source-resource": "false"})
        self.assertEqual(ec2["Condition"]["StringEquals"]["ec2:Owner"], "493765492819")

    def test_kms_access_has_only_known_source_keys_and_service_constraints(self):
        for statement in self.statements:
            if not any(a.startswith("kms:") for a in items(statement["Action"])):
                continue
            self.assertTrue(set(items(statement["Resource"])) <= KEYS)
            if "kms:CreateGrant" in items(statement["Action"]):
                self.assertEqual(statement["Condition"]["Bool"]["kms:GrantIsForAWSResource"], "true")
            elif "kms:DescribeKey" not in items(statement["Action"]):
                self.assertEqual(statement["Condition"]["StringEquals"]["kms:ViaService"],
                                 "ec2.us-east-1.amazonaws.com")

    def test_recovery_point_tagging_is_account_and_region_scoped(self):
        statements = self.statements_for("backup:TagResource")
        self.assertEqual(len(statements), 1)
        self.assertEqual(statements[0]["Resource"],
                         "arn:aws:backup:us-east-1:493765492819:recovery-point:*")

    def test_template_discloses_unencrypted_ebs_and_recovery_limits(self):
        text = TEMPLATE.read_text()
        self.assertIn("volumes are unencrypted", text)
        self.assertIn("this vault does not encrypt them", text)
        self.assertIn("crash-consistent", text)
        self.assertIn("not regional disaster recovery", text)


if __name__ == "__main__":
    unittest.main()
