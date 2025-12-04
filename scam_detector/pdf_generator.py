"""
PDF report generation for scam detection results
"""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class PDFGenerator:
    """Generates PDF reports from analysis results"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize PDF generator
        
        Args:
            output_dir: Directory to save PDF reports (defaults to report_res/)
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "report_res"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "reportlab is required for PDF generation. Install it with: pip install reportlab"
            )
    
    def generate_pdf(self, report: Dict[str, Any], results_df=None, filename: Optional[str] = None, model_name: Optional[str] = None, run_time_seconds: Optional[float] = None) -> Path:
        """
        Generate PDF report from analysis results
        
        Args:
            report: Report dictionary from ReportGenerator
            results_df: Optional DataFrame with detailed results
            filename: Optional custom filename
            model_name: Optional model name used for analysis
            run_time_seconds: Optional total run time in seconds
            
        Returns:
            Path to saved PDF file
        """
        base_name = (filename or "scam_analysis_report").replace('.pdf', '')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_name = f"{base_name}_{timestamp}.pdf"
        filepath = self.output_dir / final_name
        
        # Create PDF document
        doc = SimpleDocTemplate(str(filepath), pagesize=letter,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define professional tech color scheme
        HEADER_BG = colors.HexColor('#1F3A93')  # Deep blue
        HEADER_TEXT = colors.HexColor('#FFFFFF')  # White
        ROW_BG = colors.HexColor('#EAF2FB')  # Very light blue
        BORDER_COLOR = colors.HexColor('#C0D3EB')  # Light blue-gray
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=20
        )
        
        # White text style for table headers
        header_text_style = ParagraphStyle(
            'HeaderText',
            parent=styles['Normal'],
            fontSize=11,
            textColor=HEADER_TEXT,
            fontName='Helvetica-Bold'
        )
        
        # Title
        elements.append(Paragraph("Job Scam Detection Analysis Report", title_style))
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        
        # Add model and run time information
        metadata_info = []
        if model_name:
            metadata_info.append(f"Model: {model_name}")
        if run_time_seconds is not None:
            if run_time_seconds < 60:
                metadata_info.append(f"Run Time: {run_time_seconds:.1f} seconds")
            else:
                minutes = int(run_time_seconds // 60)
                seconds = run_time_seconds % 60
                metadata_info.append(f"Run Time: {minutes}m {seconds:.1f}s")
        
        if metadata_info:
            elements.append(Paragraph(" | ".join(metadata_info), styles['Normal']))
        
        elements.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        elements.append(Paragraph("Executive Summary", heading_style))
        summary = report.get('summary', {})
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Complaints Analyzed', str(summary.get('total_complaints', 0))],
            ['High Risk Complaints', f"{summary.get('high_risk_complaints', 0)} ({summary.get('high_risk_percentage', 0):.1f}%)"],
            ['Average Risk Score', f"{summary.get('average_risk_score', 0):.1f}%"],
            ['Median Risk Score', f"{summary.get('median_risk_score', 0):.1f}%"],
            ['High Risk Threshold', f"≥{summary.get('high_risk_threshold', 70)}%"]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HEADER_BG),
            ('TEXTCOLOR', (0, 0), (-1, 0), HEADER_TEXT),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), ROW_BG),
            ('GRID', (0, 0), (-1, -1), 1, BORDER_COLOR)
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Top Red Flags by Category (aligned with framework)
        top_red_flags_by_category = report.get('top_red_flags_by_category', {})
        if top_red_flags_by_category:
            elements.append(Paragraph("Top Red Flags by Category", heading_style))
            
            # Framework category names for display
            category_display_names = {
                'communication': 'Communication',
                'financial': 'Financial',
                'job_posting': 'Job Posting',
                'hiring_process': 'Hiring Process',
                'work_activity': 'Work Activity'
            }
            
            for category, flags_dict in top_red_flags_by_category.items():
                if flags_dict:
                    category_name = category_display_names.get(category, category.replace('_', ' ').title())
                    elements.append(Paragraph(f"{category_name}", styles['Heading2']))
                    
                    flags_data = [[Paragraph("Red Flag", header_text_style), Paragraph("Occurrences", header_text_style)]]
                    for flag, count in list(flags_dict.items())[:5]:  # Top 5 per category
                        flag_para = Paragraph(flag, styles['Normal'])
                        flags_data.append([flag_para, str(count)])
                    
                    flags_table = Table(flags_data, colWidths=[4*inch, 1*inch])
                    flags_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), HEADER_BG),
                        ('TEXTCOLOR', (0, 0), (-1, 0), HEADER_TEXT),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 11),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), ROW_BG),
                        ('GRID', (0, 0), (-1, -1), 1, BORDER_COLOR),
                        ('LEFTPADDING', (0, 0), (-1, -1), 6),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                        ('TOPPADDING', (0, 0), (-1, -1), 6),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
                    ]))
                    elements.append(flags_table)
                    elements.append(Spacer(1, 0.2*inch))
            
            elements.append(Spacer(1, 0.2*inch))
        
        # Fallback: Overall Top Red Flags (if categorized flags not available)
        top_red_flags = report.get('top_red_flags', {})
        if top_red_flags and not top_red_flags_by_category:
            elements.append(Paragraph("Top Red Flags", heading_style))
            flags_data = [[Paragraph("Red Flag", header_text_style), Paragraph("Occurrences", header_text_style)]]
            for flag, count in list(top_red_flags.items())[:10]:
                flag_para = Paragraph(flag, styles['Normal'])
                flags_data.append([flag_para, str(count)])
            
            flags_table = Table(flags_data, colWidths=[4*inch, 1*inch])
            flags_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HEADER_BG),
                ('TEXTCOLOR', (0, 0), (-1, 0), HEADER_TEXT),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), ROW_BG),
                ('GRID', (0, 0), (-1, -1), 1, BORDER_COLOR),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
            ]))
            elements.append(flags_table)
            elements.append(Spacer(1, 0.3*inch))
        
        # Top Vulnerability Factors
        vulnerability_factors = report.get('top_vulnerability_factors', {})
        if vulnerability_factors:
            elements.append(Paragraph("Top Vulnerability Factors", heading_style))
            factors_data = [[Paragraph("Vulnerability Factor", header_text_style), Paragraph("Occurrences", header_text_style)]]
            for factor, count in list(vulnerability_factors.items())[:10]:
                factor_para = Paragraph(factor, styles['Normal'])
                factors_data.append([factor_para, str(count)])
            
            factors_table = Table(factors_data, colWidths=[4*inch, 1*inch])
            factors_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HEADER_BG),
                ('TEXTCOLOR', (0, 0), (-1, 0), HEADER_TEXT),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), ROW_BG),
                ('GRID', (0, 0), (-1, -1), 1, BORDER_COLOR),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
            ]))
            elements.append(factors_table)
            elements.append(Spacer(1, 0.3*inch))
        
        # Scam Type Distribution
        scam_types = report.get('scam_type_distribution', {})
        if scam_types:
            elements.append(Paragraph("Scam Type Distribution", heading_style))
            types_data = [[Paragraph("Scam Type", header_text_style), Paragraph("Count", header_text_style)]]
            for scam_type, count in list(scam_types.items())[:10]:
                # Use Paragraph for text wrapping in long scam type names
                type_para = Paragraph(scam_type, styles['Normal'])
                types_data.append([type_para, str(count)])
            
            types_table = Table(types_data, colWidths=[4*inch, 1*inch])
            types_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HEADER_BG),
                ('TEXTCOLOR', (0, 0), (-1, 0), HEADER_TEXT),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),  # Align to top for multi-line cells
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), ROW_BG),
                ('GRID', (0, 0), (-1, -1), 1, BORDER_COLOR),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
            ]))
            elements.append(types_table)
            elements.append(Spacer(1, 0.3*inch))
        
        # Build PDF
        doc.build(elements)
        
        return filepath

