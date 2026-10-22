from gateway.tee.enclave_wheelhouse import _tag_is_compatible


def test_python312_wheel_tags_accept_current_and_older_abi3():
    assert _tag_is_compatible("cp312-cp312-manylinux2014_x86_64")
    assert _tag_is_compatible("cp311-abi3-manylinux2014_x86_64")
    assert _tag_is_compatible("py3-none-any")


def test_python312_wheel_tags_reject_newer_or_wrong_platform():
    assert not _tag_is_compatible("cp313-abi3-manylinux2014_x86_64")
    assert not _tag_is_compatible("cp312-cp312-manylinux2014_aarch64")
    assert not _tag_is_compatible("cp312-cp312-macosx_11_0_x86_64")
