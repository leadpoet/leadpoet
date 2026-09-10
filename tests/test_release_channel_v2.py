"""Gateway-only immutable release channel contract."""
import copy
import json
import pytest
from gateway.tee import release_channel_v2
from gateway.tee.release_channel_v2 import (
    SCHEMA_VERSION, ReleaseChannelV2Error, build_release_channel_v2,
    build_release_lineage_v2, install_release_channel_v2,
    publish_release_channel_v2, validate_prior_release_channel_v2,
    validate_release_channel_v2,
)
from gateway.tee.release_manifest_v2 import BUILD_EVIDENCE_SCHEMA_VERSION, build_release_manifest
from gateway.tee.release_lineage_v2 import ReleaseLineageV2Error
from gateway.tee.topology import ROLE_SPECS, topology_hash
from leadpoet_canonical.attested_v2 import sha256_json

COMMIT="1"*40
def _hash(c): return "sha256:"+c*64
def _manifest(commit=COMMIT):
 rows=[]
 for i,(role,spec) in enumerate(sorted(ROLE_SPECS.items())):
  c="abcdef0123456789"[i]
  fixed={"commit_sha":commit,"pcr0":c*96,"normalized_image_hash":_hash(c),"eif_hash":_hash(c),"source_manifest_hash":_hash("2"),"build_identity_hash":_hash(c),"execution_manifest_hash":_hash(c),"dependency_lock_hash":_hash("3"),"dockerfile_hash":_hash("4"),"topology_hash":topology_hash()}
  for domain in ("gateway","validator"):
   for ordinal in (1,2,3): rows.append({"schema_version":BUILD_EVIDENCE_SCHEMA_VERSION,"builder_domain":domain,"builder_id":domain+"-parent","build_ordinal":ordinal,"physical_role":role,"service_role":spec["service_role"],**fixed})
 return build_release_manifest(rows,acceptance_signer_pubkey_hash=_hash("f"))
class Body:
 def __init__(self,v): self.v=v
 def read(self): return self.v
class S3:
 def __init__(self): self.objects={}; self.puts=[]
 def get_object(self,**kw): return {"Body":Body(self.objects[(kw["Bucket"],kw["Key"])])}
 def put_object(self,**kw): self.objects[(kw["Bucket"],kw["Key"])]=kw["Body"]; self.puts.append(kw)

def test_current_channel_contains_only_gateway_authority():
 channel=build_release_channel_v2(gateway_release_manifest=_manifest())
 assert channel["schema_version"]==SCHEMA_VERSION
 assert set(channel)=={"schema_version","commit_sha","gateway_release_manifest","channel_hash"}
 assert validate_release_channel_v2(channel,expected_commit=COMMIT)==channel
 assert set(build_release_lineage_v2([channel],current_commit=COMMIT)["releases"][COMMIT]["roles"])==set(ROLE_SPECS)
 bad=copy.deepcopy(channel); bad["validator_release_manifest"]={}
 with pytest.raises(ReleaseChannelV2Error,match="fields"): validate_release_channel_v2(bad)

def test_historical_validator_field_is_hash_covered_but_not_authority():
 gateway=_manifest(); body={"schema_version":"leadpoet.attested_release_channel.v2","commit_sha":COMMIT,"gateway_release_manifest":gateway,"validator_release_manifest":{"opaque":"historical"}}
 old={**body,"channel_hash":sha256_json(body)}
 assert validate_prior_release_channel_v2(old)["validator_release_manifest"]=={"opaque":"historical"}
 tampered=copy.deepcopy(old); tampered["validator_release_manifest"]["opaque"]="changed"
 with pytest.raises(ReleaseChannelV2Error,match="hash"): validate_prior_release_channel_v2(tampered)


def _retired_validator_lineage():
 current = build_release_channel_v2(gateway_release_manifest=_manifest("1" * 40))
 historical = build_release_channel_v2(gateway_release_manifest=_manifest("2" * 40))
 lineage = build_release_lineage_v2(
  [current, historical], current_commit="1" * 40
 )
 for commit, release in lineage["releases"].items():
  template = next(iter(release["roles"].values()))
  release["roles"]["validator_weights"] = dict(template)
  if commit == "2" * 40:
   release["roles"]["gateway_autoresearch"] = dict(template)
 body = {key: value for key, value in lineage.items() if key != "lineage_hash"}
 return {**body, "lineage_hash": sha256_json(body)}


def test_installed_prior_lineage_discards_retired_physical_roles():
 prior = _retired_validator_lineage()
 projected = release_channel_v2._project_installed_prior_release_lineage_v2(
  prior
 )
 assert set(projected["releases"]["1" * 40]["roles"]) == set(ROLE_SPECS)
 assert set(projected["releases"]["2" * 40]["roles"]) == set(ROLE_SPECS)
 assert projected["releases"]["1" * 40]["channel_hash"] == (
  prior["releases"]["1" * 40]["channel_hash"]
 )
 assert release_channel_v2._project_installed_prior_release_lineage_v2(
  projected
 ) == projected


def test_installed_prior_lineage_discards_historical_autoresearch_only():
 prior = _retired_validator_lineage()
 for release in prior["releases"].values():
  release["roles"].pop("validator_weights")
 body = {key: value for key, value in prior.items() if key != "lineage_hash"}
 prior["lineage_hash"] = sha256_json(body)
 projected = release_channel_v2._project_installed_prior_release_lineage_v2(
  prior
 )
 assert all(
  set(release["roles"]) == set(ROLE_SPECS)
  for release in projected["releases"].values()
 )


def test_installed_prior_lineage_rejects_hash_and_retired_role_binding_drift():
 prior = _retired_validator_lineage()
 with pytest.raises(ReleaseLineageV2Error, match="hash differs"):
  release_channel_v2._project_installed_prior_release_lineage_v2(
   {**prior, "lineage_hash": _hash("9")}
  )

 drifted = copy.deepcopy(prior)
 drifted["releases"]["2" * 40]["roles"]["validator_weights"][
  "commit_sha"
 ] = "3" * 40
 body = {key: value for key, value in drifted.items() if key != "lineage_hash"}
 drifted["lineage_hash"] = sha256_json(body)
 with pytest.raises(ReleaseLineageV2Error, match="expectation is invalid"):
  release_channel_v2._project_installed_prior_release_lineage_v2(drifted)

def test_publish_is_immutable_object_locked_and_install_is_gateway_only(tmp_path):
 s3=S3(); channel=build_release_channel_v2(gateway_release_manifest=_manifest())
 published=publish_release_channel_v2(channel,bucket="bucket",s3_client=s3)
 assert published["commit_sha"]==COMMIT
 assert s3.puts[0]["ObjectLockMode"]=="COMPLIANCE" and s3.puts[0]["IfNoneMatch"]=="*"
 output=tmp_path/"gateway.json"; install_release_channel_v2(channel,expected_commit=COMMIT,gateway_output=output)
 assert json.loads(output.read_text())==channel["gateway_release_manifest"]
