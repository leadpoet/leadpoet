from typing import Any, Dict, List, Optional, Tuple

class AxonInfo:
    #  Axon serving block.
    block: int
    #  Axon version
    version: int
    #  Axon u128 encoded ip address of type v6 or v4.
    ip: int
    #  Axon u16 encoded port.
    port: int
    #  Axon ip type, 4 for ipv4 and 6 for ipv6.
    ip_type: int
    #  Axon protocol. TCP, UDP, other.
    protocol: int
    #  Axon proto placeholder 1.
    placeholder1: int
    #  Axon proto placeholder 2.
    placeholder2: int

    @staticmethod
    def decode(encoded: bytes) -> "AxonInfo":
        pass
    @staticmethod
    def decode_option(encoded: bytes) -> Optional["AxonInfo"]:
        pass
    @staticmethod
    def decode_vec(encoded: bytes) -> List["AxonInfo"]:
        pass

class PrometheusInfo:
    block: int
    # Prometheus version.
    version: int
    #  Prometheus u128 encoded ip address of type v6 or v4.
    ip: int
    # Prometheus u16 encoded port.
    port: int
    # Prometheus ip type, 4 for ipv4 and 6 for ipv6.
    ip_type: int

    @staticmethod
    def decode(encoded: bytes) -> "PrometheusInfo":
        pass
    @staticmethod
    def decode_option(encoded: bytes) -> Optional["PrometheusInfo"]:
        pass
    @staticmethod
    def decode_vec(encoded: bytes) -> List["PrometheusInfo"]:
        pass

class NeuronInfo:
    hotkey: bytes
    coldkey: bytes
    uid: int
    netuid: int
    active: bool
    axon_info: AxonInfo
    prometheus_info: PrometheusInfo
    stake: List[
        Tuple[bytes, int]
    ]  # map of coldkey to stake on this neuron/hotkey (includes delegations)
    rank: int
    emission: int
    incentive: int
    consensus: int
    trust: int
    validator_trust: int
    dividends: int
    last_update: int
    validator_permit: bool
    weights: List[Tuple[int, int]]  # Vec of (uid, weight)
    bonds: List[Tuple[int, int]]  # Vec of (uid, bond)
    pruning_score: int

    @staticmethod
    def decode(encoded: bytes) -> "NeuronInfo":
        pass
    @staticmethod
    def decode_option(encoded: bytes) -> Optional["NeuronInfo"]:
        pass
    @staticmethod
    def decode_vec(encoded: bytes) -> List["NeuronInfo"]:
        pass

class NeuronInfoLite:
    hotkey: bytes
    coldkey: bytes
    uid: int
    netuid: int
    active: bool
    axon_info: AxonInfo
    prometheus_info: PrometheusInfo
    stake: List[
        Tuple[bytes, int]
    ]  # map of coldkey to stake on this neuron/hotkey (includes delegations)
    rank: int
    emission: int
    incentive: int
    consensus: int
    trust: int
    validator_trust: int
    dividends: int
    last_update: int
    validator_permit: bool
    # has no weights or bonds
    pruning_score: int

    @staticmethod
    def decode(encoded: bytes) -> "NeuronInfoLite":
        pass
    @staticmethod
    def decode_option(encoded: bytes) -> Optional["NeuronInfoLite"]:
        pass
    @staticmethod
    def decode_vec(encoded: bytes) -> List["NeuronInfoLite"]:
        pass

class SubnetIdentity:
    subnet_name: bytes  # TODO: or List[int] ??
    # The github repository associated with the chain identity
    github_repo: bytes
    # The subnet's contact
    subnet_contact: bytes

    @staticmethod
    def decode(encoded: bytes) -> "SubnetIdentity":
        pass
    @staticmethod
    def decode_option(encoded: bytes) -> Optional["SubnetIdentity"]:
        pass
    @staticmethod
    def decode_vec(encoded: bytes) -> List["SubnetIdentity"]:
        pass

class SubnetInfo:
    netuid: int
    rho: int
    kappa: int
    difficulty: int
    immunity_period: int
    max_allowed_validators: int
    min_allowed_weights: int
    max_weights_limit: int
    scaling_law_power: int
    subnetwork_n: int
    max_allowed_uids: int
    blocks_since_last_step: int
    tempo: int
    network_modality: int
    network_connect: List[List[int]]  # List[[int, int]]
    emission_values: int
    burn: int
    owner: bytes

    @staticmethod
    def decode(encoded: bytes) -> "SubnetInfo":
        pass
    @staticmethod
    def decode_option(encoded: bytes) -> Optional["SubnetInfo"]:
        pass
    @staticmethod
    def decode_vec(encoded: bytes) -> List["SubnetInfo"]:
        pass
    @staticmethod
    def decode_vec_option(encoded: bytes) -> List[Optional["SubnetInfo"]]:
        pass

class SubnetInfoV2:
    netuid: int
    rho: int
    kappa: int
    difficulty: int
    immunity_period: int
    max_allowed_validators: int
    min_allowed_weights: int
    max_weights_limit: int
    scaling_law_power: int
    subnetwork_n: int
    max_allowed_uids: int
    blocks_since_last_step: int
    tempo: int
    network_modality: int
    network_connect: List[List[int]]  # List[[int, int]]
    emission_values: int
    burn: int
    owner: bytes
    identity: Optional[SubnetIdentity]

    @staticmethod
    def decode(encoded: bytes) -> "SubnetInfoV2":
        pass
    @staticmethod
    def decode_option(encoded: bytes) -> Optional["SubnetInfoV2"]:
        pass
    @staticmethod
    def decode_vec(encoded: bytes) -> List["SubnetInfoV2"]:
        pass
    @staticmethod
    def decode_vec_option(encoded: bytes) -> List[Optional["SubnetInfoV2"]]:
        pass

class SubnetHyperparameters:
    rho: int
    kappa: int
    immunity_period: int
    min_allowed_weights: int
    max_weights_limit: int
    tempo: int
    min_difficulty: int
    max_difficulty: int
    weights_version: int
    weights_rate_limit: int
    adjustment_interval: int
    activity_cutoff: int
    registration_allowed: bool
    target_regs_per_interval: int
    min_burn: int
    max_burn: int
    bonds_moving_avg: int
    max_regs_per_block: int
    serving_rate_limit: int
    max_validators: int
    adjustment_alpha: int
    difficulty: int
    commit_reveal_weights_interval: int
    commit_reveal_weights_enabled: bool
    alpha_high: int
    alpha_low: int
    liquid_alpha_enabled: bool

    @staticmethod
    def decode(encoded: bytes) -> "SubnetHyperparameters":
        pass
    @staticmethod
    def decode_option(encoded: bytes) -> Optional["SubnetHyperparameters"]:
        pass
    @staticmethod
    def decode_vec(encoded: bytes) -> List["SubnetHyperparameters"]:
        pass

class StakeInfo:
    hotkey: bytes
    coldkey: bytes
    stake: int

    @staticmethod
    def decode(encoded: bytes) -> "StakeInfo":
        pass
    @staticmethod
    def decode_option(encoded: bytes) -> Optional["StakeInfo"]:
        pass
    @staticmethod
    def decode_vec(encoded: bytes) -> List["StakeInfo"]:
        pass
    @staticmethod
    def decode_vec_tuple_vec(encoded: bytes) -> List[Tuple[bytes, List["StakeInfo"]]]:
        pass

class DelegateInfo:
    delegate_ss58: bytes
    take: int
    nominators: List[Tuple[bytes, int]]  # map of nominator_ss58 to stake amount
    owner_ss58: bytes
    registrations: List[int]  # Vec of netuid this delegate is registered on
    validator_permits: List[int]  # Vec of netuid this delegate has validator permit on
    return_per_1000: (
        int  # Delegators current daily return per 1000 TAO staked minus take fee
    )
    total_daily_return: int

    @staticmethod
    def decode(encoded: bytes) -> "DelegateInfo":
        pass
    @staticmethod
    def decode_option(encoded: bytes) -> Optional["DelegateInfo"]:
        pass
    @staticmethod
    def decode_vec(encoded: bytes) -> List["DelegateInfo"]:
        pass
    @staticmethod
    def decode_delegated(encoded: bytes) -> List[Tuple["DelegateInfo", int]]:
        pass

class MetadataV15:
    """
    MetadataV15 is the 15th version-style of metadata for the chain.
    It contains information about all the chain types, including the type signatures
    of the Runtime API functions.

    Example:
    >>> import bittensor, bt_decode, scalecodec
    >>> sub = bittensor.subtensor()
    >>> v15_int = scalecodec.U32()
    >>> v15_int.value = 15
    >>> metadata_rpc_result = sub.substrate.rpc_request("state_call", [
    ...     "Metadata_metadata_at_version",
    ...     v15_int.encode().to_hex(),
    ...     sub.substrate.get_chain_finalised_head()
    ])
    >>> metadata_option_hex_str = metadata_rpc_result['result']
    >>> metadata_option_bytes = bytes.fromhex(metadata_option_hex_str[2:])
    >>> metadata_v15 = bt_decode.MetadataV15.decode_from_metadata_option(metadata_option_bytes)
    >>> print(metadata_v15.to_json())
    """

    @staticmethod
    def decode_from_metadata_option(encoded_metadata_v15: bytes) -> "MetadataV15":
        pass
    def to_json(self) -> str:
        """
        Returns a JSON representation of the metadata.
        """
        pass
    def value(self) -> Dict[str, Any]:
        pass

class PortableRegistry:
    """
    PortableRegistry is a portable for of the chains registry that
    can be used to serialize and deserialize the registry to and from JSON.

    Example:
    >>> import bittensor, bt_decode, scalecodec
    >>> sub = bittensor.subtensor()
    >>> v15_int = scalecodec.U32()
    >>> v15_int.value = 15
    >>> metadata_rpc_result = sub.substrate.rpc_request("state_call", [
    ...     "Metadata_metadata_at_version",
    ...     v15_int.encode().to_hex(),
    ...     sub.substrate.get_chain_finalised_head()
    ])
    >>> metadata_option_hex_str = metadata_rpc_result['result']
    >>> metadata_option_bytes = bytes.fromhex(metadata_option_hex_str[2:])
    >>> metadata_v15 = bt_decode.MetadataV15.decode_from_metadata_option(metadata_option_bytes)
    >>> bt_decode.PortableRegistry.from_metadata_v15( metadata_v15 )
    """

    registry: str  # JSON encoded PortableRegistry

    @staticmethod
    def from_json(json_str: str) -> "PortableRegistry":
        pass
    @staticmethod
    def from_metadata_v15(metadata_v15: MetadataV15) -> "PortableRegistry":
        pass

def decode(
    type_string: str, portable_registry: PortableRegistry, encoded: bytes
) -> Any:
    pass

def encode(
    type_string: str, portable_registry: PortableRegistry, to_encode: Any
) -> list[int]:
    """
    Encode a python object to bytes.

    Returns a list of integers representing the encoded bytes.

    Example:
    >>> import bittensor as bt
    >>> res = bt.decode.encode("u128", bt.decode.PortableRegistry.from_json(...), 1234567890)
    >>> res
    [210, 2, 150, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    >>> bytes(res).hex()
    'd2029649000000000000000000000000'
    """
    pass
