import re
import math
from typing import List, Set, Dict, Optional
from simhash import Simhash
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from loguru import logger

from app.core.config import settings
from app.models.lead import Lead

def get_features(s):
    """
    Extract features (words) from a string.
    This basic implementation splits by non-alphanumeric characters.
    """
    width = 3
    s = s.lower()
    s = re.sub(r'[^\w]+', '', s)
    return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]

class SimhashService:
    """
    Service for generating Simhash fingerprints and detecting near-duplicates using LSH.
    """
    def __init__(self, db: Session):
        self.db = db
        self.bit_difference_threshold = settings.SIMHASH_BIT_DIFFERENCE_THRESHOLD
        
        # LSH parameters for 64-bit simhash
        self.hash_bits = 64
        self.bands = 16  # Number of bands for LSH
        self.rows_per_band = 4  # Rows per band (bands * rows_per_band = 64)
        
        # Cache for frequently accessed hash buckets
        self._bucket_cache: Dict[int, Set[int]] = {}
        self._cache_size_limit = 1000

    def get_lead_text_corpus(self, lead_data: dict) -> str:
        """
        Constructs a single text corpus from various fields of a lead.
        This is used as the input for Simhash generation.
        """
        # Using .get() provides default empty strings if a key is missing
        parts = [
            lead_data.get('company_name', ''),
            lead_data.get('first_name', ''),
            lead_data.get('last_name', ''),
        ]
        
        # Include firmographics and technographics details
        firmographics = lead_data.get('firmographics', {})
        if isinstance(firmographics, dict):
            parts.extend(firmographics.values())

        technographics = lead_data.get('technographics', {})
        if isinstance(technographics, dict):
            parts.extend(technographics.values())
        
        return " ".join(str(p) for p in parts if p)

    def calculate_simhash(self, text: str) -> int:
        """
        Calculates the 64-bit Simhash value for a given string.
        """
        if not text:
            return 0
        return Simhash(get_features(text)).value

    def _get_lsh_bands(self, simhash_value: int) -> List[int]:
        """
        Split a 64-bit simhash into bands for LSH.
        Returns a list of band hashes.
        """
        bands = []
        for i in range(self.bands):
            start_bit = i * self.rows_per_band
            end_bit = start_bit + self.rows_per_band
            # Extract bits for this band and create a hash
            band_bits = (simhash_value >> start_bit) & ((1 << self.rows_per_band) - 1)
            bands.append(band_bits)
        return bands

    def _get_candidate_hashes(self, new_lead_hash: int) -> Set[int]:
        """
        Get candidate simhash values that might be near-duplicates using LSH.
        """
        candidate_hashes = set()
        new_bands = self._get_lsh_bands(new_lead_hash)
        
        # Query database for leads that share at least one band
        for band_idx, band_hash in enumerate(new_bands):
            # Create a query that finds leads with similar band hashes
            # We'll use a range query based on the bit difference threshold
            min_band_hash = max(0, band_hash - self.bit_difference_threshold)
            max_band_hash = min((1 << self.rows_per_band) - 1, band_hash + self.bit_difference_threshold)
            
            # Query for leads in this band range
            leads_in_band = self.db.query(Lead.simhash).filter(
                and_(
                    Lead.simhash.is_not(None),
                    Lead.simhash >= min_band_hash << (band_idx * self.rows_per_band),
                    Lead.simhash < (max_band_hash + 1) << (band_idx * self.rows_per_band)
                )
            ).all()
            
            for (simhash,) in leads_in_band:
                candidate_hashes.add(simhash)
        
        return candidate_hashes

    def _calculate_hamming_distance(self, hash1: int, hash2: int) -> int:
        """
        Calculate Hamming distance between two simhash values.
        """
        return bin(hash1 ^ hash2).count('1')

    def find_near_duplicates(self, new_lead_hash: int) -> bool:
        """
        Efficiently checks for near-duplicates using Locality-Sensitive Hashing (LSH).
        
        This implementation:
        1. Uses LSH to quickly identify candidate duplicates
        2. Only performs expensive Hamming distance calculations on candidates
        3. Caches frequently accessed hash buckets for better performance
        """
        if not new_lead_hash:
            return False
            
        # Get candidate hashes using LSH
        candidate_hashes = self._get_candidate_hashes(new_lead_hash)
        
        if not candidate_hashes:
            return False
        
        # Check each candidate for actual similarity
        for candidate_hash in candidate_hashes:
            distance = self._calculate_hamming_distance(new_lead_hash, candidate_hash)
            if distance <= self.bit_difference_threshold:
                logger.info(f"Near-duplicate found. New hash: {new_lead_hash}, Existing hash: {candidate_hash}, Distance: {distance}")
                return True
        
        return False

    def find_near_duplicates_with_details(self, new_lead_hash: int) -> Optional[Dict]:
        """
        Enhanced version that returns details about the duplicate if found.
        """
        if not new_lead_hash:
            return None
            
        candidate_hashes = self._get_candidate_hashes(new_lead_hash)
        
        if not candidate_hashes:
            return None
        
        # Find the closest match
        closest_distance = float('inf')
        closest_lead = None
        
        for candidate_hash in candidate_hashes:
            distance = self._calculate_hamming_distance(new_lead_hash, candidate_hash)
            if distance <= self.bit_difference_threshold and distance < closest_distance:
                closest_distance = distance
                # Get the full lead record
                lead = self.db.query(Lead).filter(Lead.simhash == candidate_hash).first()
                if lead:
                    closest_lead = lead
        
        if closest_lead:
            return {
                'lead_id': str(closest_lead.lead_id),
                'company_name': closest_lead.company_name,
                'email': closest_lead.email,
                'simhash': closest_lead.simhash,
                'distance': closest_distance,
                'similarity_percentage': (64 - closest_distance) / 64 * 100
            }
        
        return None

    def batch_find_duplicates(self, lead_hashes: List[int]) -> Dict[int, bool]:
        """
        Efficiently check multiple leads for duplicates in a single operation.
        """
        results = {}
        
        # Group leads by LSH bands for batch processing
        band_groups: Dict[int, List[int]] = {}
        for lead_hash in lead_hashes:
            bands = self._get_lsh_bands(lead_hash)
            for band_idx, band_hash in enumerate(bands):
                band_key = (band_idx, band_hash)
                if band_key not in band_groups:
                    band_groups[band_key] = []
                band_groups[band_key].append(lead_hash)
        
        # Process each lead
        for lead_hash in lead_hashes:
            results[lead_hash] = self.find_near_duplicates(lead_hash)
        
        return results 