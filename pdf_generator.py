"""
Professional PDF Report Generator for Fine-Tuning Calculator
===========================================================

This module provides professional PDF report generation capabilities
for the fine-tuning cost calculator, including executive summaries,
detailed analysis reports, and comparison charts.
"""

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("⚠️ ReportLab not installed. Install with: pip install reportlab")

import datetime
import os
from typing import List, Dict, Any
import json


class PDFReportGenerator:
    """Professional PDF report generator for fine-tuning calculations"""

    def __init__(self):
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF generation. Install with: pip install reportlab")

        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles for professional reports"""

        # Executive Summary Title
        self.styles.add(ParagraphStyle(
            name='ExecutiveTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        # Section Headers
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold'
        ))

        # Subsection Headers
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.blue,
            fontName='Helvetica-Bold'
        ))

        # Key Metrics
        self.styles.add(ParagraphStyle(
            name='KeyMetric',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            textColor=colors.black,
            fontName='Helvetica-Bold'
        ))

        # Cost Highlight
        self.styles.add(ParagraphStyle(
            name='CostHighlight',
            parent=self.styles['Normal'],
            fontSize=14,
            spaceAfter=6,
            textColor=colors.green,
            fontName='Helvetica-Bold',
            alignment=TA_CENTER
        ))

    def generate_executive_summary(self, calculations: List[Dict], output_filename: str) -> str:
        """Generate executive summary PDF for CEO presentation"""

        doc = SimpleDocTemplate(output_filename, pagesize=A4)
        story = []

        # Title
        title = Paragraph("Fine-Tuning Cost Analysis", self.styles['ExecutiveTitle'])
        story.append(title)
        story.append(Spacer(1, 20))

        # Header info
        header_info = f"""
        <b>Executive Summary</b><br/>
        Prepared by: ML Engineering Team<br/>
        Date: {datetime.datetime.now().strftime('%B %d, %Y')}<br/>
        Analysis Period: {len(calculations)} scenarios evaluated
        """
        story.append(Paragraph(header_info, self.styles['Normal']))
        story.append(Spacer(1, 20))

        if not calculations:
            story.append(Paragraph("No calculations available for analysis.", self.styles['Normal']))
            doc.build(story)
            return output_filename

        # Key Findings
        story.append(Paragraph("Key Findings", self.styles['SectionHeader']))

        total_cost = sum(calc['result']['total_cost'] for calc in calculations)
        avg_cost = total_cost / len(calculations)
        min_cost = min(calc['result']['total_cost'] for calc in calculations)
        max_cost = max(calc['result']['total_cost'] for calc in calculations)
        avg_time = sum(calc['result']['wallclock_hours'] for calc in calculations) / len(calculations)

        findings_data = [
            ["Metric", "Value"],
            ["Total Scenarios Analyzed", f"{len(calculations)}"],
            ["Total Estimated Cost", f"${total_cost:,.2f}"],
            ["Average Cost per Scenario", f"${avg_cost:,.2f}"],
            ["Cost Range", f"${min_cost:.2f} - ${max_cost:.2f}"],
            ["Average Training Time", f"{avg_time:.1f} hours"],
        ]

        findings_table = Table(findings_data, colWidths=[3*inch, 2*inch])
        findings_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(findings_table)
        story.append(Spacer(1, 20))

        # Cost Breakdown Chart (if multiple calculations)
        if len(calculations) > 1:
            story.append(Paragraph("Cost Analysis", self.styles['SectionHeader']))

            # Create cost comparison table
            cost_data = [["Scenario", "Model", "Provider", "Total Cost", "Training Time"]]
            for i, calc in enumerate(calculations, 1):
                input_data = calc['input']
                result_data = calc['result']
                cost_data.append([
                    f"Scenario {i}",
                    input_data.get('model_config', {}).get('name', 'Unknown'),
                    input_data.get('provider', 'Unknown'),
                    f"${result_data['total_cost']:.2f}",
                    f"{result_data['wallclock_hours']:.1f}h"
                ])

            cost_table = Table(cost_data, colWidths=[1*inch, 1.5*inch, 1.5*inch, 1*inch, 1*inch])
            cost_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
            ]))

            story.append(cost_table)
            story.append(Spacer(1, 20))

        # Recommendations
        story.append(Paragraph("Strategic Recommendations", self.styles['SectionHeader']))

        recommendations = [
            "• <b>Cost Optimization:</b> Consider QLoRA for up to 60% cost reduction while maintaining quality",
            "• <b>Model Selection:</b> 7B-13B models sufficient for most business applications",
            "• <b>Provider Strategy:</b> Lambda Labs and RunPod offer competitive pricing for development",
            "• <b>Timeline Planning:</b> Budget 2-7 days for training depending on model size",
            "• <b>Budget Allocation:</b> Reserve 20% contingency for unexpected training iterations"
        ]

        for rec in recommendations:
            story.append(Paragraph(rec, self.styles['Normal']))
            story.append(Spacer(1, 6))

        story.append(Spacer(1, 20))

        # Next Steps
        story.append(Paragraph("Next Steps", self.styles['SectionHeader']))

        next_steps = [
            "1. <b>Budget Approval:</b> Secure funding based on cost estimates",
            "2. <b>Provider Selection:</b> Choose optimal cloud provider and configuration",
            "3. <b>Pilot Project:</b> Begin with smallest viable model for proof of concept",
            "4. <b>Performance Monitoring:</b> Track actual vs. estimated costs and performance",
            "5. <b>Scale Planning:</b> Develop roadmap for production deployment"
        ]

        for step in next_steps:
            story.append(Paragraph(step, self.styles['Normal']))
            story.append(Spacer(1, 6))

        # Footer
        story.append(Spacer(1, 30))
        footer_text = f"<i>Report generated by Fine-Tuning Cost Calculator v1.0 on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        story.append(Paragraph(footer_text, self.styles['Normal']))

        # Build PDF
        doc.build(story)
        return output_filename

    def generate_detailed_report(self, calculation: Dict, output_filename: str) -> str:
        """Generate detailed technical report for a single calculation"""

        doc = SimpleDocTemplate(output_filename, pagesize=A4)
        story = []

        calc_name = calculation.get('name', 'Unnamed Calculation')
        input_data = calculation['input']
        result_data = calculation['result']

        # Title
        title = Paragraph(f"Fine-Tuning Analysis: {calc_name}", self.styles['ExecutiveTitle'])
        story.append(title)
        story.append(Spacer(1, 20))

        # Configuration Details
        story.append(Paragraph("Configuration Details", self.styles['SectionHeader']))

        config_data = [
            ["Parameter", "Value"],
            ["Model", input_data.get('model_config', {}).get('name', 'Unknown')],
            ["Model Size", input_data.get('model_config', {}).get('size', 'Unknown')],
            ["Cloud Provider", input_data.get('provider', 'Unknown')],
            ["GPU Type", input_data.get('gpu_type', 'Unknown')],
            ["Number of GPUs", str(input_data.get('num_gpus', 'Unknown'))],
            ["Training Method", input_data.get('training_method', 'Unknown').upper()],
            ["Training Examples", f"{input_data.get('num_examples', 0):,}"],
            ["Tokens per Example", str(input_data.get('tokens_per_example', 'Unknown'))],
            ["Epochs", str(input_data.get('epochs', 'Unknown'))],
            ["Batch Size", str(input_data.get('batch_size', 'Unknown'))],
        ]

        config_table = Table(config_data, colWidths=[2.5*inch, 2.5*inch])
        config_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(config_table)
        story.append(Spacer(1, 20))

        # Training Metrics
        story.append(Paragraph("Training Metrics", self.styles['SectionHeader']))

        metrics_data = [
            ["Metric", "Value"],
            ["Total Tokens", f"{result_data['total_tokens']:,}"],
            ["Effective Batch Size", str(result_data['effective_batch_size'])],
            ["Steps per Epoch", f"{result_data['steps_per_epoch']:,}"],
            ["Total Training Steps", f"{result_data['total_steps']:,}"],
            ["GPU Hours (Total)", f"{result_data['gpu_hours']:.1f}"],
            ["Wallclock Time", f"{result_data['wallclock_hours']:.1f} hours"],
        ]

        if result_data['wallclock_hours'] > 24:
            days = result_data['wallclock_hours'] / 24
            metrics_data.append(["Wallclock Time (Days)", f"{days:.1f} days"])

        metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(metrics_table)
        story.append(Spacer(1, 20))

        # Cost Breakdown
        story.append(Paragraph("Cost Analysis", self.styles['SectionHeader']))

        # Highlight total cost
        total_cost_text = f"Total Estimated Cost: ${result_data['total_cost']:.2f}"
        story.append(Paragraph(total_cost_text, self.styles['CostHighlight']))
        story.append(Spacer(1, 15))

        cost_data = [
            ["Cost Component", "Amount", "Percentage"],
            ["Compute Cost", f"${result_data['compute_cost']:.2f}",
             f"{(result_data['compute_cost']/result_data['total_cost']*100):.1f}%"],
            ["Storage Cost", f"${result_data['storage_cost']:.2f}",
             f"{(result_data['storage_cost']/result_data['total_cost']*100):.1f}%"],
            ["Data Transfer", f"${result_data['data_transfer_cost']:.2f}",
             f"{(result_data['data_transfer_cost']/result_data['total_cost']*100):.1f}%"],
        ]

        if result_data['additional_fees'] > 0:
            cost_data.append(["Additional Fees", f"${result_data['additional_fees']:.2f}",
                            f"{(result_data['additional_fees']/result_data['total_cost']*100):.1f}%"])

        cost_data.append(["TOTAL", f"${result_data['total_cost']:.2f}", "100.0%"])

        cost_table = Table(cost_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        cost_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -2), colors.beige),
            ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(cost_table)
        story.append(Spacer(1, 20))

        # Efficiency Metrics
        story.append(Paragraph("Efficiency Metrics", self.styles['SectionHeader']))

        efficiency_data = [
            ["Metric", "Value"],
            ["Cost per Token", f"${result_data['cost_per_token']:.6f}"],
            ["Cost per Example", f"${result_data['cost_per_example']:.4f}"],
            ["Tokens per Dollar", f"{1/result_data['cost_per_token']:,.0f}"],
            ["Examples per Dollar", f"{1/result_data['cost_per_example']:,.1f}"],
        ]

        efficiency_table = Table(efficiency_data, colWidths=[2.5*inch, 2.5*inch])
        efficiency_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(efficiency_table)
        story.append(Spacer(1, 30))

        # Footer
        footer_text = f"<i>Report generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Calculation date: {calculation.get('timestamp', 'Unknown')[:10]}</i>"
        story.append(Paragraph(footer_text, self.styles['Normal']))

        # Build PDF
        doc.build(story)
        return output_filename

    def generate_comparison_report(self, comparison_data: Dict, output_filename: str) -> str:
        """Generate comparison report for multiple scenarios"""

        doc = SimpleDocTemplate(output_filename, pagesize=A4)
        story = []

        scenarios = comparison_data['scenarios']

        # Title
        title = Paragraph("Fine-Tuning Scenario Comparison", self.styles['ExecutiveTitle'])
        story.append(title)
        story.append(Spacer(1, 20))

        # Summary
        summary = comparison_data['comparison_summary']
        story.append(Paragraph("Comparison Summary", self.styles['SectionHeader']))

        summary_text = f"""
        <b>Analysis Overview:</b><br/>
        • Total scenarios compared: {summary['total_scenarios']}<br/>
        • Cost range: ${summary['cost_range'][0]:.2f} - ${summary['cost_range'][1]:.2f}<br/>
        • Time range: {summary['time_range'][0]:.1f} - {summary['time_range'][1]:.1f} hours<br/>
        """
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 20))

        # Detailed Comparison Table
        story.append(Paragraph("Detailed Comparison", self.styles['SectionHeader']))

        table_data = [["Rank", "Model", "Provider", "Method", "Cost", "Time", "Cost/Example"]]

        for i, scenario in enumerate(scenarios, 1):
            input_data = scenario['input']
            result = scenario['result']
            table_data.append([
                str(i),
                input_data.get('model_config', {}).get('name', 'Unknown')[:15],
                input_data.get('provider', 'Unknown')[:12],
                input_data.get('training_method', 'Unknown').upper(),
                f"${result.total_cost:.2f}",
                f"{result.wallclock_hours:.1f}h",
                f"${result.cost_per_example:.4f}"
            ])

        comparison_table = Table(table_data, colWidths=[0.5*inch, 1.5*inch, 1*inch, 0.8*inch, 0.8*inch, 0.7*inch, 1*inch])
        comparison_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            # Highlight best cost (first row)
            ('BACKGROUND', (0, 1), (-1, 1), colors.lightgreen),
        ]))

        story.append(comparison_table)
        story.append(Spacer(1, 20))

        # Best Options
        story.append(Paragraph("Recommended Options", self.styles['SectionHeader']))

        best_cost = comparison_data['best_cost']
        best_time = comparison_data['best_time']

        recommendations_text = f"""
        <b>Most Cost-Effective:</b> {best_cost['summary']['model']} on {best_cost['summary']['provider']}<br/>
        • Total cost: ${best_cost['result'].total_cost:.2f}<br/>
        • Training time: {best_cost['result'].wallclock_hours:.1f} hours<br/><br/>

        <b>Fastest Training:</b> {best_time['summary']['model']} on {best_time['summary']['provider']}<br/>
        • Training time: {best_time['result'].wallclock_hours:.1f} hours<br/>
        • Total cost: ${best_time['result'].total_cost:.2f}<br/>
        """

        story.append(Paragraph(recommendations_text, self.styles['Normal']))
        story.append(Spacer(1, 30))

        # Footer
        footer_text = f"<i>Comparison report generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        story.append(Paragraph(footer_text, self.styles['Normal']))

        # Build PDF
        doc.build(story)
        return output_filename


def install_reportlab():
    """Install ReportLab if not available"""
    if not REPORTLAB_AVAILABLE:
        try:
            import subprocess
            import sys
            print("Installing ReportLab for PDF generation...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
            print("✅ ReportLab installed successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to install ReportLab: {e}")
            return False
    return True


if __name__ == "__main__":
    # Test PDF generation
    print("Testing PDF generation...")

    if not REPORTLAB_AVAILABLE:
        print("ReportLab not available. Install with: pip install reportlab")
    else:
        generator = PDFReportGenerator()

        # Test data
        test_calculation = {
            "name": "Test Calculation",
            "timestamp": datetime.datetime.now().isoformat(),
            "input": {
                "model_config": {"name": "Llama 2 7B", "size": "7B"},
                "provider": "aws_sagemaker",
                "gpu_type": "ml.g5.xlarge",
                "num_gpus": 1,
                "training_method": "qlora",
                "num_examples": 10000,
                "tokens_per_example": 512,
                "epochs": 3,
                "batch_size": 4
            },
            "result": {
                "total_tokens": 15360000,
                "effective_batch_size": 4,
                "steps_per_epoch": 2500,
                "total_steps": 7500,
                "gpu_hours": 307.2,
                "wallclock_hours": 307.2,
                "compute_cost": 309.04,
                "storage_cost": 1.53,
                "data_transfer_cost": 0.90,
                "additional_fees": 0.0,
                "total_cost": 311.47,
                "cost_per_token": 0.000020,
                "cost_per_example": 0.0311
            }
        }

        try:
            filename = generator.generate_detailed_report(test_calculation, "test_report.pdf")
            print(f"✅ Test PDF generated: {filename}")
        except Exception as e:
            print(f"❌ PDF generation failed: {e}")
