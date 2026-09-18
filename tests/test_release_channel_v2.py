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
from gateway.tee.release_lineage_v2 import (
    build_compact_release_lineage_boot_verifier_v2,
)
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

def test_prior_channel_accepts_v3_only_and_rejects_validator_payload():
 channel=build_release_channel_v2(gateway_release_manifest=_manifest())
 assert validate_prior_release_channel_v2(channel)==channel
 old=copy.deepcopy(channel); old["validator_release_manifest"]={"opaque":"retired"}
 with pytest.raises(ReleaseChannelV2Error,match="fields"):
  validate_prior_release_channel_v2(old)


def _installed_two_role_lineage():
 current = build_release_channel_v2(gateway_release_manifest=_manifest("1" * 40))
 historical = build_release_channel_v2(gateway_release_manifest=_manifest("2" * 40))
 lineage = build_release_lineage_v2(
  [current, historical], current_commit="1" * 40
 )
 for release in lineage["releases"].values():
  template = next(iter(release["roles"].values()))
  release["roles"]["gateway_scoring"] = {
   **template,
   "pcr0": "e" * 96,
   "build_manifest_hash": _hash("e"),
  }
 body = {key: value for key, value in lineage.items() if key != "lineage_hash"}
 return {**body, "lineage_hash": sha256_json(body)}


def test_installed_prior_lineage_accepts_exact_two_role_history():
 prior = _installed_two_role_lineage()
 projected = release_channel_v2._project_installed_prior_release_lineage_v2(
  prior
 )
 assert all(
  set(release["roles"]) == {"gateway_coordinator", "gateway_scoring"}
  for release in projected["releases"].values()
 )
 assert projected == prior


def test_installed_prior_lineage_rejects_hash_and_scoring_binding_drift():
 prior = _installed_two_role_lineage()
 with pytest.raises(ReleaseLineageV2Error, match="hash differs"):
  release_channel_v2._project_installed_prior_release_lineage_v2(
   {**prior, "lineage_hash": _hash("9")}
  )

 drifted = copy.deepcopy(prior)
 drifted["releases"]["2" * 40]["roles"]["gateway_scoring"][
  "commit_sha"
 ] = "3" * 40
 body = {key: value for key, value in drifted.items() if key != "lineage_hash"}
 drifted["lineage_hash"] = sha256_json(body)
 with pytest.raises(ReleaseLineageV2Error, match="expectation is invalid"):
  release_channel_v2._project_installed_prior_release_lineage_v2(drifted)


def test_fetch_retains_projected_installed_ancestors_for_generic_boot_proof(
 monkeypatch, tmp_path
):
 current_commit = "3" * 40
 gateway_path = tmp_path / "gateway.json"
 gateway_path.write_text(json.dumps(_manifest(current_commit)))
 prior_path = tmp_path / "lineage.json"
 prior_path.write_text(json.dumps(_installed_two_role_lineage()))
 monkeypatch.setenv("LEADPOET_LOCAL_RELEASE_COMMIT_SHA", current_commit)
 monkeypatch.setenv("LEADPOET_LOCAL_GATEWAY_RELEASE", str(gateway_path))
 monkeypatch.setenv("LEADPOET_LOCAL_PRIOR_RELEASE_LINEAGE", str(prior_path))

 lineage = release_channel_v2.fetch_release_lineage_v2(
  bucket="unused",
  current_commit=current_commit,
  allowed_commits=(current_commit, "1" * 40, "2" * 40),
  required_commits=(current_commit,),
 )

 assert set(lineage["releases"]) == {
  current_commit, "1" * 40, "2" * 40
 }
 assert set(lineage["releases"][current_commit]["roles"]) == set(ROLE_SPECS)
 for commit in ("1" * 40, "2" * 40):
  assert set(lineage["releases"][commit]["roles"]) == {
   "gateway_coordinator", "gateway_scoring"
  }
 prior_role = lineage["releases"]["2" * 40]["roles"][
  "gateway_coordinator"
 ]
 identity = {
  "physical_role": "gateway_coordinator",
  **prior_role,
 }
 verifier = build_compact_release_lineage_boot_verifier_v2(
  lineage, boot_verifier=lambda value, **_kwargs: value
 )
 assert verifier(identity) == identity


def test_fetch_rejects_installed_commit_outside_bounded_git_ancestry(
 monkeypatch, tmp_path
):
 current_commit = "3" * 40
 gateway_path = tmp_path / "gateway.json"
 gateway_path.write_text(json.dumps(_manifest(current_commit)))
 prior_path = tmp_path / "lineage.json"
 prior_path.write_text(json.dumps(_installed_two_role_lineage()))
 monkeypatch.setenv("LEADPOET_LOCAL_RELEASE_COMMIT_SHA", current_commit)
 monkeypatch.setenv("LEADPOET_LOCAL_GATEWAY_RELEASE", str(gateway_path))
 monkeypatch.setenv("LEADPOET_LOCAL_PRIOR_RELEASE_LINEAGE", str(prior_path))

 with pytest.raises(ReleaseChannelV2Error, match="non-ancestor"):
  release_channel_v2.fetch_release_lineage_v2(
   bucket="unused",
   current_commit=current_commit,
   allowed_commits=(current_commit, "1" * 40),
   required_commits=(current_commit,),
  )

def test_publish_is_immutable_object_locked_and_install_is_gateway_only(tmp_path):
 s3=S3(); channel=build_release_channel_v2(gateway_release_manifest=_manifest())
 published=publish_release_channel_v2(channel,bucket="bucket",s3_client=s3)
 assert published["commit_sha"]==COMMIT
 assert s3.puts[0]["ObjectLockMode"]=="COMPLIANCE" and s3.puts[0]["IfNoneMatch"]=="*"
 output=tmp_path/"gateway.json"; install_release_channel_v2(channel,expected_commit=COMMIT,gateway_output=output)
 assert json.loads(output.read_text())==channel["gateway_release_manifest"]
