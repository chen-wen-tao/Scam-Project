#!/usr/bin/env python3
"""
Visualization script for job scam detection results
Creates charts and graphs to visualize scam patterns and trends
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, Any
import numpy as np

class ScamVisualizer:
    """Create visualizations for scam detection results"""
    
    def __init__(self, results_file: str, report_file: str = None):
        """
        Initialize visualizer with results data
        
        Args:
            results_file: Path to CSV file with analysis results
            report_file: Optional path to JSON report file
        """
        self.results_df = pd.read_csv(results_file)
        self.report = None
        
        # Use risk_score (Gemini's scam_probability) or fallback to overall_risk_score for backward compatibility
        self.risk_score_col = 'risk_score' if 'risk_score' in self.results_df.columns else 'overall_risk_score'
        
        if report_file:
            with open(report_file, 'r') as f:
                self.report = json.load(f)
    
    def plot_risk_distribution(self, save_path: str = None):
        """Plot distribution of risk scores"""
        plt.figure(figsize=(12, 6))
        
        # Create histogram
        plt.subplot(1, 2, 1)
        plt.hist(self.results_df[self.risk_score_col], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Risk Score')
        plt.ylabel('Number of Complaints')
        plt.title('Distribution of Risk Scores')
        plt.grid(True, alpha=0.3)
        
        # Create box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(self.results_df[self.risk_score_col], vert=True)
        plt.ylabel('Risk Score')
        plt.title('Risk Score Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Risk distribution plot saved to: {save_path}")
        
        plt.close()  # Close figure to free memory
    
    def plot_red_flags_heatmap(self, save_path: str = None):
        """Create heatmap of red flags from Gemini analysis"""
        # Extract red flags from Gemini analysis (much more meaningful than rule-based)
        red_flag_data = []
        
        for idx, row in self.results_df.iterrows():
            analysis = row['gemini_analysis']
            if isinstance(analysis, str):
                import ast
                analysis = ast.literal_eval(analysis)
            
            if isinstance(analysis, dict) and 'red_flags' in analysis:
                red_flags = analysis['red_flags']
                if isinstance(red_flags, list):
                    for flag in red_flags:
                        red_flag_data.append({
                            'flag': flag,
                            'risk_score': row[self.risk_score_col]
                        })
        
        if not red_flag_data:
            print("No red flag data found for heatmap")
            return
        
        red_flag_df = pd.DataFrame(red_flag_data)
        
        # Create frequency table of top red flags
        top_flags = red_flag_df['flag'].value_counts().head(20)
        
        # Create a simple bar chart instead of heatmap since we don't have categories
        plt.figure(figsize=(15, 8))
        top_flags.plot(kind='barh', color='coral')
        plt.title('Top Red Flags from Gemini Analysis')
        plt.xlabel('Frequency')
        plt.ylabel('Red Flag')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Red flags chart saved to: {save_path}")
        
        plt.tight_layout()
        plt.close()  # Close figure to free memory
    
    def plot_scam_types(self, save_path: str = None):
        """Plot distribution of scam types"""
        if not self.report or 'scam_type_distribution' not in self.report:
            print("No scam type data available")
            return
        
        scam_types = self.report['scam_type_distribution']
        
        # Create pie chart
        plt.figure(figsize=(10, 8))
        labels = list(scam_types.keys())[:8]  # Top 8 scam types
        sizes = list(scam_types.values())[:8]
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('Distribution of Scam Types')
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Scam types plot saved to: {save_path}")
        
        plt.close()  # Close figure to free memory
    
    def plot_risk_vs_text_length(self, save_path: str = None):
        """Plot risk score vs text length correlation"""
        plt.figure(figsize=(10, 6))
        
        plt.scatter(self.results_df['text_length'], self.results_df[self.risk_score_col], 
                   alpha=0.6, color='red')
        
        # Add trend line
        z = np.polyfit(self.results_df['text_length'], self.results_df[self.risk_score_col], 1)
        p = np.poly1d(z)
        plt.plot(self.results_df['text_length'], p(self.results_df[self.risk_score_col]), 
                "r--", alpha=0.8, linewidth=2)
        
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Risk Score')
        plt.title('Risk Score vs Text Length')
        plt.grid(True, alpha=0.3)
        
        # Calculate correlation
        correlation = self.results_df['text_length'].corr(self.results_df[self.risk_score_col])
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Risk vs text length plot saved to: {save_path}")
        
        plt.close()  # Close figure to free memory
    
    def plot_top_red_flags(self, save_path: str = None, top_n: int = 15):
        """Plot top red flags as horizontal bar chart"""
        if not self.report or 'top_red_flags' not in self.report:
            print("No red flag data available")
            return
        
        top_flags = dict(list(self.report['top_red_flags'].items())[:top_n])
        
        plt.figure(figsize=(12, 8))
        
        flags = list(top_flags.keys())
        counts = list(top_flags.values())
        
        # Create horizontal bar chart
        y_pos = np.arange(len(flags))
        plt.barh(y_pos, counts, color='lightcoral')
        
        plt.yticks(y_pos, flags)
        plt.xlabel('Frequency')
        plt.title(f'Top {top_n} Red Flags Found in Complaints')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, v in enumerate(counts):
            plt.text(v + 0.1, i, str(v), va='center')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Top red flags plot saved to: {save_path}")
        
        plt.tight_layout()
        plt.close()  # Close figure to free memory
    
    def create_dashboard(self, save_path: str = None):
        """Create a comprehensive dashboard with multiple plots"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Job Scam Detection Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Risk Score Distribution
        axes[0, 0].hist(self.results_df['overall_risk_score'], bins=15, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Risk Score Distribution')
        axes[0, 0].set_xlabel('Risk Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Risk vs Text Length
        axes[0, 1].scatter(self.results_df['text_length'], self.results_df['overall_risk_score'], 
                          alpha=0.6, color='red')
        axes[0, 1].set_title('Risk Score vs Text Length')
        axes[0, 1].set_xlabel('Text Length')
        axes[0, 1].set_ylabel('Risk Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. High Risk Complaints Timeline (if we have timestamps)
        if 'timestamp' in self.results_df.columns:
            high_risk = self.results_df[self.results_df[self.risk_score_col] >= 70]
            if len(high_risk) > 0:
                axes[1, 0].hist(high_risk[self.risk_score_col], bins=10, alpha=0.7, color='darkred')
                axes[1, 0].set_title('High Risk Complaints (Score ≥ 70)')
                axes[1, 0].set_xlabel('Risk Score')
                axes[1, 0].set_ylabel('Frequency')
            else:
                axes[1, 0].text(0.5, 0.5, 'No High Risk Complaints', ha='center', va='center', 
                               transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('High Risk Complaints')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Timestamp Data', ha='center', va='center', 
                           transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('High Risk Complaints')
        
        # 4. Summary Statistics
        stats_text = f"""
        Total Complaints: {len(self.results_df)}
        Avg Risk Score: {self.results_df[self.risk_score_col].mean():.1f}
        High Risk (≥70): {len(self.results_df[self.results_df[self.risk_score_col] >= 70])}
        Max Risk Score: {self.results_df[self.risk_score_col].max()}
        Min Risk Score: {self.results_df[self.risk_score_col].min()}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
        axes[1, 1].set_title('Summary Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Dashboard saved to: {save_path}")
        
        plt.close()  # Close figure to free memory
    
    def generate_all_visualizations(self, output_dir: str = "visualizations"):
        """Generate all visualizations and save to output directory"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating visualizations...")
        
        # Risk distribution
        self.plot_risk_distribution(f"{output_dir}/risk_distribution.png")
        
        # Red flags heatmap
        self.plot_red_flags_heatmap(f"{output_dir}/red_flags_heatmap.png")
        
        # Scam types
        self.plot_scam_types(f"{output_dir}/scam_types.png")
        
        # Risk vs text length
        self.plot_risk_vs_text_length(f"{output_dir}/risk_vs_text_length.png")
        
        # Top red flags
        self.plot_top_red_flags(f"{output_dir}/top_red_flags.png")
        
        # Dashboard
        self.create_dashboard(f"{output_dir}/dashboard.png")
        
        print(f"\n✅ All visualizations saved to: {output_dir}/")
        print(f"   - risk_distribution.png")
        print(f"   - red_flags_heatmap.png")
        print(f"   - scam_types.png")
        print(f"   - risk_vs_text_length.png")
        print(f"   - top_red_flags.png")
        print(f"   - dashboard.png")

def main():
    """Main function to generate visualizations"""
    from pathlib import Path
    from scam_detector.config import DEFAULT_OUTPUT_DIR, DEFAULT_RESULTS_CSV, DEFAULT_REPORT_JSON
    
    # Use detect_res/ folder for results
    results_file = str(DEFAULT_OUTPUT_DIR / DEFAULT_RESULTS_CSV)
    report_file = str(DEFAULT_OUTPUT_DIR / DEFAULT_REPORT_JSON)
    
    try:
        visualizer = ScamVisualizer(results_file, report_file)
        visualizer.generate_all_visualizations()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the main analysis script first to generate results.")
        print(f"Expected files:")
        print(f"  - {results_file}")
        print(f"  - {report_file}")
    except Exception as e:
        print(f"Error generating visualizations: {e}")

if __name__ == "__main__":
    main()
