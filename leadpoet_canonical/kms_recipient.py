"""Strict AWS KMS Nitro recipient ciphertext decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from cryptography.hazmat.primitives import hashes, padding as symmetric_padding
from cryptography.hazmat.primitives.asymmetric import padding as asymmetric_padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


MAX_RECIPIENT_ENVELOPE_BYTES = 64 * 1024
_OID_ENVELOPED_DATA = "1.2.840.113549.1.7.3"
_OID_DATA = "1.2.840.113549.1.7.1"
_OID_RSAES_OAEP = "1.2.840.113549.1.1.7"
_OID_MGF1 = "1.2.840.113549.1.1.8"
_OID_SHA256 = "2.16.840.1.101.3.4.2.1"
_OID_AES_256_CBC = "2.16.840.1.101.3.4.1.42"


class KMSRecipientCiphertextError(ValueError):
    """The KMS recipient response is malformed or cannot be unwrapped."""


@dataclass(frozen=True)
class _TLV:
    tag_class: int
    tag_number: int
    constructed: bool
    value: bytes
    children: Tuple["_TLV", ...]


def _parse_tlv(data: bytes, offset: int, *, depth: int = 0) -> tuple[_TLV, int]:
    if depth > 16 or offset >= len(data):
        raise KMSRecipientCiphertextError("recipient CMS nesting is invalid")
    first = data[offset]
    offset += 1
    tag_class = first >> 6
    constructed = bool(first & 0x20)
    tag_number = first & 0x1F
    if tag_number == 0x1F:
        raise KMSRecipientCiphertextError("recipient CMS high tags are unsupported")
    if offset >= len(data):
        raise KMSRecipientCiphertextError("recipient CMS length is missing")
    length_byte = data[offset]
    offset += 1
    indefinite = length_byte == 0x80
    if indefinite and not constructed:
        raise KMSRecipientCiphertextError("recipient CMS primitive is indefinite")
    if indefinite:
        children = []
        while True:
            if offset + 2 > len(data):
                raise KMSRecipientCiphertextError("recipient CMS EOC is missing")
            if data[offset : offset + 2] == b"\x00\x00":
                return _TLV(tag_class, tag_number, True, b"", tuple(children)), offset + 2
            child, offset = _parse_tlv(data, offset, depth=depth + 1)
            children.append(child)
    if length_byte & 0x80:
        count = length_byte & 0x7F
        if count == 0 or count > 4 or offset + count > len(data):
            raise KMSRecipientCiphertextError("recipient CMS length is invalid")
        length = int.from_bytes(data[offset : offset + count], "big")
        if length < 128:
            raise KMSRecipientCiphertextError("recipient CMS length is non-canonical")
        offset += count
    else:
        length = length_byte
    end = offset + length
    if end > len(data):
        raise KMSRecipientCiphertextError("recipient CMS value is truncated")
    if not constructed:
        return _TLV(tag_class, tag_number, False, data[offset:end], ()), end
    children = []
    cursor = offset
    while cursor < end:
        child, cursor = _parse_tlv(data, cursor, depth=depth + 1)
        children.append(child)
    if cursor != end:
        raise KMSRecipientCiphertextError("recipient CMS children are invalid")
    return _TLV(tag_class, tag_number, True, b"", tuple(children)), end


def _expect(node: _TLV, tag_class: int, tag_number: int, constructed: bool) -> None:
    if (
        node.tag_class != tag_class
        or node.tag_number != tag_number
        or node.constructed is not constructed
    ):
        raise KMSRecipientCiphertextError("recipient CMS structure is invalid")


def _oid(node: _TLV) -> str:
    _expect(node, 0, 6, False)
    if not node.value:
        raise KMSRecipientCiphertextError("recipient CMS OID is empty")
    first = node.value[0]
    parts = [min(first // 40, 2), first - 40 * min(first // 40, 2)]
    value = 0
    continued = False
    for byte in node.value[1:]:
        value = (value << 7) | (byte & 0x7F)
        continued = bool(byte & 0x80)
        if not byte & 0x80:
            parts.append(value)
            value = 0
    if continued:
        raise KMSRecipientCiphertextError("recipient CMS OID is truncated")
    return ".".join(str(part) for part in parts)


def _collect_oids(node: _TLV) -> tuple[str, ...]:
    values = []
    if node.tag_class == 0 and node.tag_number == 6 and not node.constructed:
        values.append(_oid(node))
    for child in node.children:
        values.extend(_collect_oids(child))
    return tuple(values)


def _octets(node: _TLV) -> bytes:
    if node.tag_class == 0 and node.tag_number == 4 and not node.constructed:
        return node.value
    if node.tag_class == 2 and node.tag_number == 0 and not node.constructed:
        return node.value
    if not node.constructed:
        raise KMSRecipientCiphertextError("recipient CMS ciphertext is invalid")
    values = [_octets(child) for child in node.children]
    if not values:
        raise KMSRecipientCiphertextError("recipient CMS ciphertext is empty")
    return b"".join(values)


def _decrypt_cms(private_key: Any, ciphertext: bytes) -> bytes:
    root, consumed = _parse_tlv(ciphertext, 0)
    if consumed != len(ciphertext):
        raise KMSRecipientCiphertextError("recipient CMS has trailing data")
    _expect(root, 0, 16, True)
    if len(root.children) != 2 or _oid(root.children[0]) != _OID_ENVELOPED_DATA:
        raise KMSRecipientCiphertextError("recipient CMS content type is invalid")
    wrapper = root.children[1]
    _expect(wrapper, 2, 0, True)
    if len(wrapper.children) != 1:
        raise KMSRecipientCiphertextError("recipient CMS wrapper is invalid")
    enveloped = wrapper.children[0]
    _expect(enveloped, 0, 16, True)
    if len(enveloped.children) < 3:
        raise KMSRecipientCiphertextError("recipient CMS envelope is incomplete")

    recipients = enveloped.children[1]
    _expect(recipients, 0, 17, True)
    if len(recipients.children) != 1:
        raise KMSRecipientCiphertextError("recipient CMS must have one recipient")
    recipient = recipients.children[0]
    _expect(recipient, 0, 16, True)
    if len(recipient.children) != 4:
        raise KMSRecipientCiphertextError("recipient CMS recipient is invalid")
    key_algorithm = recipient.children[2]
    _expect(key_algorithm, 0, 16, True)
    algorithm_oids = _collect_oids(key_algorithm)
    if (
        not algorithm_oids
        or algorithm_oids[0] != _OID_RSAES_OAEP
        or _OID_MGF1 not in algorithm_oids
        or algorithm_oids.count(_OID_SHA256) < 2
    ):
        raise KMSRecipientCiphertextError("recipient CMS key algorithm is invalid")
    encrypted_key = _octets(recipient.children[3])
    if len(encrypted_key) != private_key.key_size // 8:
        raise KMSRecipientCiphertextError("recipient CMS RSA ciphertext size is invalid")

    encrypted_content_info = enveloped.children[2]
    _expect(encrypted_content_info, 0, 16, True)
    if len(encrypted_content_info.children) != 3:
        raise KMSRecipientCiphertextError("recipient CMS content is invalid")
    if _oid(encrypted_content_info.children[0]) != _OID_DATA:
        raise KMSRecipientCiphertextError("recipient CMS payload type is invalid")
    content_algorithm = encrypted_content_info.children[1]
    _expect(content_algorithm, 0, 16, True)
    if (
        len(content_algorithm.children) != 2
        or _oid(content_algorithm.children[0]) != _OID_AES_256_CBC
    ):
        raise KMSRecipientCiphertextError("recipient CMS content algorithm is invalid")
    iv_node = content_algorithm.children[1]
    _expect(iv_node, 0, 4, False)
    if len(iv_node.value) != 16:
        raise KMSRecipientCiphertextError("recipient CMS IV is invalid")
    encrypted_content = _octets(encrypted_content_info.children[2])
    if not encrypted_content or len(encrypted_content) % 16:
        raise KMSRecipientCiphertextError("recipient CMS AES ciphertext is invalid")

    try:
        aes_key = private_key.decrypt(
            encrypted_key,
            asymmetric_padding.OAEP(
                mgf=asymmetric_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        if len(aes_key) != 32:
            raise KMSRecipientCiphertextError("recipient CMS AES key is invalid")
        decryptor = Cipher(
            algorithms.AES(aes_key),
            modes.CBC(iv_node.value),
        ).decryptor()
        padded = decryptor.update(encrypted_content) + decryptor.finalize()
        unpadder = symmetric_padding.PKCS7(128).unpadder()
        return unpadder.update(padded) + unpadder.finalize()
    except KMSRecipientCiphertextError:
        raise
    except Exception as exc:
        raise KMSRecipientCiphertextError(
            "recipient CMS unwrap failed"
        ) from exc


def decrypt_kms_recipient_ciphertext(private_key: Any, ciphertext: bytes) -> bytes:
    """Decrypt raw RSA ingress or AWS KMS CMS EnvelopedData ciphertext."""

    value = bytes(ciphertext)
    if not value or len(value) > MAX_RECIPIENT_ENVELOPE_BYTES:
        raise KMSRecipientCiphertextError("recipient ciphertext is outside limit")
    if len(value) == private_key.key_size // 8:
        try:
            return private_key.decrypt(
                value,
                asymmetric_padding.OAEP(
                    mgf=asymmetric_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )
        except Exception as exc:
            raise KMSRecipientCiphertextError(
                "raw recipient unwrap failed"
            ) from exc
    return _decrypt_cms(private_key, value)
