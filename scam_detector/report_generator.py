"""
Report generation for scam detection results
"""

import pandas as pd
import re
from typing import Dict, Any


def _normalize_scam_type_name(scam_type: str) -> str:
    """
    Normalize scam type names to merge similar variations.
    
    The "/" in scam type names means "and" (both characteristics present).
    This function normalizes spacing, capitalization, and merges similar variations.
    
    Examples:
    - "Fake Check Scam / Money Mule" -> "Fake Check / Money Mule Scam"
    - "Fake Check/Money Mule Scam" -> "Fake Check / Money Mule Scam"
    - "Fake Check Scam / Money Mule Scam" -> "Fake Check / Money Mule Scam"
    
    Args:
        scam_type: Original scam type name
        
    Returns:
        Normalized scam type name
    """
    if not isinstance(scam_type, str):
        return str(scam_type)
    
    # Normalize spacing around "/" - standardize to " / "
    normalized = re.sub(r'\s*/\s*', ' / ', scam_type.strip())
    
    # Normalize common variations
    # "Fake Check Scam / Money Mule" -> "Fake Check / Money Mule Scam"
    normalized = re.sub(r'Fake Check Scam\s*/\s*Money Mule(\s*Scam)?', 'Fake Check / Money Mule Scam', normalized, flags=re.IGNORECASE)
    
    # "Fake Check / Money Mule Scam" (already normalized)
    # "Fake Check/Money Mule Scam" -> "Fake Check / Money Mule Scam"
    normalized = re.sub(r'Fake Check\s*/\s*Money Mule(\s*Scam)?', 'Fake Check / Money Mule Scam', normalized, flags=re.IGNORECASE)
    
    # "Fake Check Scam / Money Mule Scam" -> "Fake Check / Money Mule Scam"
    normalized = re.sub(r'Fake Check Scam\s*/\s*Money Mule Scam', 'Fake Check / Money Mule Scam', normalized, flags=re.IGNORECASE)
    
    # "Fake Job Offer/Money Mule" -> "Fake Job Offer / Money Mule Scam"
    normalized = re.sub(r'Fake Job Offer\s*/\s*Money Mule(\s*Scam)?', 'Fake Job Offer / Money Mule Scam', normalized, flags=re.IGNORECASE)
    
    # "Fake Job Offer/Check Scam" -> "Fake Job Offer / Check Scam"
    normalized = re.sub(r'Fake Job Offer\s*/\s*Check(\s*Scam)?', 'Fake Job Offer / Check Scam', normalized, flags=re.IGNORECASE)
    
    # "Fake Job Offer with Fake Checks" -> "Fake Job Offer / Check Scam"
    normalized = re.sub(r'Fake Job Offer\s+with\s+Fake Checks?', 'Fake Job Offer / Check Scam', normalized, flags=re.IGNORECASE)
    
    # "Fake Check / Advance Fee Scam" -> "Fake Check / Advance Fee Scam" (keep as is)
    # "Fake Check Scam" -> keep as is (different from combined types)
    
    # Normalize capitalization: title case for consistency
    # Split by " / " to preserve the separator
    parts = normalized.split(' / ')
    normalized_parts = []
    for part in parts:
        # Title case each part, but preserve common acronyms
        part_normalized = part.title()
        # Fix common acronyms
        part_normalized = re.sub(r'\bMlm\b', 'MLM', part_normalized, flags=re.IGNORECASE)
        part_normalized = re.sub(r'\bSsn\b', 'SSN', part_normalized, flags=re.IGNORECASE)
        normalized_parts.append(part_normalized)
    
    normalized = ' / '.join(normalized_parts)
    
    return normalized


class ReportGenerator:
    """Generates comprehensive analysis reports"""
    
    @staticmethod
    def generate_report(results_df: pd.DataFrame, high_risk_threshold: int = 70) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report
        
        Args:
            results_df: DataFrame with analysis results
            high_risk_threshold: Threshold for high risk (default: 70)
            
        Returns:
            Report dictionary
        """
        total_complaints = len(results_df)
        
        # Use risk_score (which is Gemini's scam_probability)
        risk_score_col = 'risk_score' if 'risk_score' in results_df.columns else 'overall_risk_score'
        
        # Risk distribution
        risk_distribution = results_df[risk_score_col].value_counts().sort_index()
        
        # High risk complaints based on Gemini's scam_probability
        high_risk_mask = results_df[risk_score_col] >= high_risk_threshold
        high_risk = results_df[high_risk_mask].copy() if high_risk_mask.any() else pd.DataFrame()
        
        # Extract red flags from Gemini analysis, grouped by category (aligned with framework)
        # Framework categories: communication, financial, job_posting, hiring_process, work_activity
        red_flags_by_category = {
            'communication': [],
            'financial': [],
            'job_posting': [],
            'hiring_process': [],
            'work_activity': []
        }
        all_red_flags = []  # Keep for backward compatibility
        
        for analysis in results_df['gemini_analysis']:
            try:
                # Handle both dict and string representations
                if isinstance(analysis, str):
                    import json
                    try:
                        analysis = json.loads(analysis)
                    except (json.JSONDecodeError, ValueError):
                        import ast
                        try:
                            analysis = ast.literal_eval(analysis)
                        except (ValueError, SyntaxError):
                            continue
                
                if isinstance(analysis, dict) and 'red_flags' in analysis:
                    red_flags = analysis['red_flags']
                    # red_flags should be a dict with categories (aligned with framework)
                    if isinstance(red_flags, dict):
                        # Extract flags by category
                        for category, flags in red_flags.items():
                            # Normalize category name (handle variations)
                            normalized_category = category.lower().replace('_', '').replace('-', '')
                            # Map to framework categories
                            if 'communication' in normalized_category:
                                target_category = 'communication'
                            elif 'financial' in normalized_category:
                                target_category = 'financial'
                            elif 'job' in normalized_category and 'posting' in normalized_category:
                                target_category = 'job_posting'
                            elif 'hiring' in normalized_category or 'process' in normalized_category:
                                target_category = 'hiring_process'
                            elif 'work' in normalized_category or 'activity' in normalized_category:
                                target_category = 'work_activity'
                            else:
                                target_category = category  # Use original if no match
                            
                            if isinstance(flags, list):
                                # Filter out invalid entries
                                valid_flags = [f for f in flags if isinstance(f, str) and f.strip() and f.strip() not in ['{', '}', '{}', '[]', '']]
                                if target_category in red_flags_by_category:
                                    red_flags_by_category[target_category].extend(valid_flags)
                                all_red_flags.extend(valid_flags)
                            elif isinstance(flags, str):
                                # Filter out invalid string entries
                                if flags.strip() and flags.strip() not in ['{', '}', '{}', '[]', '']:
                                    if target_category in red_flags_by_category:
                                        red_flags_by_category[target_category].append(flags.strip())
                                    all_red_flags.append(flags.strip())
                    elif isinstance(red_flags, list):
                        # Fallback: if red_flags is a list, add to all_red_flags
                        valid_flags = [f for f in red_flags if isinstance(f, str) and f.strip() and f.strip() not in ['{', '}', '{}', '[]', '']]
                        all_red_flags.extend(valid_flags)
            except Exception as e:
                # Skip problematic entries
                continue
        
        # Count red flags by category (top 5 per category)
        top_red_flags_by_category = {}
        for category, flags in red_flags_by_category.items():
            if flags:
                flag_counts = pd.Series(flags).value_counts()
                top_red_flags_by_category[category] = flag_counts.head(5).to_dict()
        
        # Overall top red flags (for backward compatibility)
        red_flag_counts = pd.Series(all_red_flags).value_counts() if all_red_flags else pd.Series()
        
        # Scam type distribution
        scam_types = []
        for analysis in results_df['gemini_analysis']:
            try:
                # Handle both dict and string representations
                if isinstance(analysis, str):
                    import json
                    try:
                        analysis = json.loads(analysis)
                    except (json.JSONDecodeError, ValueError):
                        import ast
                        try:
                            analysis = ast.literal_eval(analysis)
                        except (ValueError, SyntaxError):
                            continue
                
                if isinstance(analysis, dict) and 'scam_type' in analysis:
                    scam_type = analysis['scam_type']
                    # scam_type can be a dict or a string
                    if isinstance(scam_type, dict):
                        # Extract subcategory or primary category name
                        type_name = scam_type.get('subcategory') or scam_type.get('primary_category')
                        
                        # If both are missing, try to get any string value from the dict
                        if not type_name:
                            # Try to find any string value in the dict
                            for key, value in scam_type.items():
                                if isinstance(value, str) and value.strip():
                                    type_name = value
                                    break
                            # If still no valid name, use "Unknown" instead of str(scam_type)
                            if not type_name:
                                type_name = "Unknown"
                        
                        # Handle case where type_name might be a list (convert to string)
                        if isinstance(type_name, list):
                            type_name = ', '.join(str(item) for item in type_name if item)
                        elif not isinstance(type_name, str):
                            type_name = "Unknown"
                        
                        # Filter out invalid entries (empty, just braces, etc.)
                        type_name = type_name.strip()
                        if type_name and type_name not in ['{', '}', '{}', '[]', '']:
                            scam_types.append(type_name)
                    elif isinstance(scam_type, str):
                        # Filter out invalid string entries
                        if scam_type.strip() and scam_type.strip() not in ['{', '}', '{}', '[]']:
                            scam_types.append(scam_type.strip())
                    elif isinstance(scam_type, list):
                        # Handle case where scam_type itself is a list
                        valid_items = [str(item) for item in scam_type if item and str(item).strip() not in ['{', '}', '{}', '[]']]
                        if valid_items:
                            scam_types.append(', '.join(valid_items))
            except Exception as e:
                # Skip problematic entries
                continue
        
        # Normalize scam type names to merge similar variations
        normalized_scam_types = []
        for scam_type in scam_types:
            normalized = _normalize_scam_type_name(scam_type)
            normalized_scam_types.append(normalized)
        
        scam_type_distribution = pd.Series(normalized_scam_types).value_counts()
        
        # Extract vulnerability factors from Gemini analysis
        all_vulnerability_factors = []
        for analysis in results_df['gemini_analysis']:
            try:
                # Handle both dict and string representations
                if isinstance(analysis, str):
                    import json
                    try:
                        analysis = json.loads(analysis)
                    except (json.JSONDecodeError, ValueError):
                        import ast
                        try:
                            analysis = ast.literal_eval(analysis)
                        except (ValueError, SyntaxError):
                            continue
                
                if isinstance(analysis, dict) and 'victim_profile' in analysis:
                    victim_profile = analysis['victim_profile']
                    if isinstance(victim_profile, dict) and 'vulnerability_factors' in victim_profile:
                        factors = victim_profile['vulnerability_factors']
                        if isinstance(factors, list):
                            valid_factors = [f for f in factors if isinstance(f, str) and f.strip() and f.strip() not in ['{', '}', '{}', '[]', '']]
                            all_vulnerability_factors.extend(valid_factors)
                        elif isinstance(factors, str):
                            if factors.strip() and factors.strip() not in ['{', '}', '{}', '[]', '']:
                                all_vulnerability_factors.append(factors.strip())
            except Exception as e:
                # Skip problematic entries
                continue
        
        vulnerability_factor_counts = pd.Series(all_vulnerability_factors).value_counts() if all_vulnerability_factors else pd.Series()
        
        report = {
            'summary': {
                'total_complaints': total_complaints,
                'high_risk_complaints': len(high_risk),
                'high_risk_percentage': (len(high_risk) / total_complaints * 100) if total_complaints > 0 else 0,
                'average_risk_score': float(results_df[risk_score_col].mean()),
                'median_risk_score': float(results_df[risk_score_col].median()),
                'high_risk_threshold': high_risk_threshold
            },
            'risk_distribution': risk_distribution.to_dict(),
            'top_red_flags': red_flag_counts.head(10).to_dict() if len(red_flag_counts) > 0 else {},  # Overall top flags (backward compatibility)
            'top_red_flags_by_category': top_red_flags_by_category,  # New: grouped by framework categories
            'top_vulnerability_factors': vulnerability_factor_counts.head(10).to_dict() if len(vulnerability_factor_counts) > 0 else {},
            'scam_type_distribution': scam_type_distribution.to_dict(),
            'high_risk_complaints': high_risk[['complaint_id', risk_score_col]].to_dict(orient='records') if len(high_risk) > 0 else []  # type: ignore
        }
        
        return report

