"""
Report generation for scam detection results
"""

import pandas as pd
from typing import Dict, Any


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
        high_risk = results_df[results_df[risk_score_col] >= high_risk_threshold]
        
        # Extract red flags from Gemini analysis (much more meaningful than rule-based)
        all_red_flags = []
        for analysis in results_df['gemini_analysis']:
            if isinstance(analysis, str):
                import ast
                analysis = ast.literal_eval(analysis)
            if isinstance(analysis, dict) and 'red_flags' in analysis:
                red_flags = analysis['red_flags']
                if isinstance(red_flags, list):
                    all_red_flags.extend(red_flags)
        
        red_flag_counts = pd.Series(all_red_flags).value_counts() if all_red_flags else pd.Series()
        
        # Scam type distribution
        scam_types = []
        for analysis in results_df['gemini_analysis']:
            if isinstance(analysis, str):
                import ast
                analysis = ast.literal_eval(analysis)
            if isinstance(analysis, dict) and 'scam_type' in analysis:
                scam_types.append(analysis['scam_type'])
        
        scam_type_distribution = pd.Series(scam_types).value_counts()
        
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
            'top_red_flags': red_flag_counts.head(10).to_dict() if len(red_flag_counts) > 0 else {},
            'scam_type_distribution': scam_type_distribution.to_dict(),
            'high_risk_complaints': high_risk[['complaint_id', risk_score_col]].to_dict('records') if len(high_risk) > 0 else []
        }
        
        return report

