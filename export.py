"""
PDF export and sharing functionality for ChemoGuard reports.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.pdfgen import canvas
from io import BytesIO
import qrcode
from PIL import Image as PILImage
from flask import current_app
from cbc import ANALYTES
from database import (
    PREDICTION_INPUT_ANALYTES,
    Prediction,
    User,
    panel_from_columns,
)
from datetime import datetime


def generate_pdf_report(prediction_id: int, user: User) -> BytesIO:
    """Generate PDF report for a prediction."""
    try:
        prediction = Prediction.query.filter_by(
            id=prediction_id,
            user_id=user.id
        ).first_or_404()
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=1  # Center
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=12
        )
        
        # Title
        elements.append(Paragraph("ChemoGuard Toxicity Risk Report", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Patient Information
        elements.append(Paragraph("Patient Information", heading_style))
        patient_info = [
            ['Patient Name:', user.full_name or user.username],
            ['Report Date:', prediction.prediction_date.strftime('%Y-%m-%d %H:%M:%S') if prediction.prediction_date else 'N/A'],
            ['Report ID:', f'CG-{prediction.id:06d}']
        ]
        patient_table = Table(patient_info, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(patient_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Input Values
        elements.append(Paragraph("Input Blood Parameters", heading_style))
        # Units are read from the stored quantities, never asserted. This
        # table used to hardcode "thousand/μL" beside whatever float was in
        # the column, so a report recorded in cells/μL printed as
        # "85000.00 thousand/μL" -- a fabricated unit, on the page a
        # clinician actually reads.
        panel = panel_from_columns(
            prediction, PREDICTION_INPUT_ANALYTES, prefix='input_'
        )
        as_reported = prediction.input_reported_quantities or {}

        input_data = [['Parameter', 'Value', 'Unit', 'As reported']]
        for key in PREDICTION_INPUT_ANALYTES:
            quantity = panel.get(key)
            if quantity is None:
                continue
            original = as_reported.get(key) or {}
            original_text = (
                f"{original['value']:g} {original['unit']}"
                if original.get('unit') is not None else '--'
            )
            input_data.append([
                ANALYTES[key].label,
                f"{quantity.value:,.2f}".rstrip('0').rstrip('.'),
                quantity.unit.display,
                original_text,
            ])

        input_table = Table(input_data, colWidths=[2.1*inch, 1.4*inch, 1.3*inch, 1.4*inch])
        input_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        elements.append(input_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Risk Predictions
        elements.append(Paragraph("Toxicity Risk Predictions", heading_style))
        
        # Risk color mapping
        risk_colors = {
            'Low': colors.HexColor('#27ae60'),
            'Medium': colors.HexColor('#f39c12'),
            'High': colors.HexColor('#e74c3c')
        }
        
        risk_data = [
            ['Toxicity Type', 'Risk Level', 'Probability (Low)', 'Probability (Medium)', 'Probability (High)'],
            [
                'Anemia',
                prediction.anemia_risk,
                f"{prediction.anemia_probability_low*100:.1f}%" if prediction.anemia_probability_low else 'N/A',
                f"{prediction.anemia_probability_medium*100:.1f}%" if prediction.anemia_probability_medium else 'N/A',
                f"{prediction.anemia_probability_high*100:.1f}%" if prediction.anemia_probability_high else 'N/A'
            ],
            [
                'Neutropenia',
                prediction.neutropenia_risk,
                f"{prediction.neutropenia_probability_low*100:.1f}%" if prediction.neutropenia_probability_low else 'N/A',
                f"{prediction.neutropenia_probability_medium*100:.1f}%" if prediction.neutropenia_probability_medium else 'N/A',
                f"{prediction.neutropenia_probability_high*100:.1f}%" if prediction.neutropenia_probability_high else 'N/A'
            ],
            [
                'Thrombocytopenia',
                prediction.thrombocytopenia_risk,
                f"{prediction.thrombocytopenia_probability_low*100:.1f}%" if prediction.thrombocytopenia_probability_low else 'N/A',
                f"{prediction.thrombocytopenia_probability_medium*100:.1f}%" if prediction.thrombocytopenia_probability_medium else 'N/A',
                f"{prediction.thrombocytopenia_probability_high*100:.1f}%" if prediction.thrombocytopenia_probability_high else 'N/A'
            ],
        ]
        
        risk_table = Table(risk_data, colWidths=[1.5*inch, 1*inch, 1.2*inch, 1.2*inch, 1.2*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        elements.append(risk_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Combined Severity
        severity_color = risk_colors.get(prediction.combined_severity, colors.HexColor('#000000'))
        elements.append(Paragraph("Combined Toxicity Severity", heading_style))
        # Get hex value - HexColor has .hex attribute, fallback to '#000000' for black
        hex_value = severity_color.hex if hasattr(severity_color, 'hex') else '#000000'
        severity_text = f"<b>Overall Risk Level: <font color='{hex_value}'>{(prediction.combined_severity or 'N/A').upper()}</font></b>"
        elements.append(Paragraph(severity_text, styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Interpretation
        elements.append(Paragraph("Interpretation", heading_style))
        interpretation = get_interpretation(prediction)
        elements.append(Paragraph(interpretation, styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Footer
        footer_text = f"Generated by ChemoGuard on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph(footer_text, ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=1
        )))
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer
    
    except Exception as e:
        raise Exception(f"Error generating PDF: {str(e)}")


def get_interpretation(prediction: Prediction) -> str:
    """Get interpretation text for the prediction."""
    interpretations = []
    
    if prediction.anemia_risk == 'High':
        interpretations.append("• <b>Anemia Risk: HIGH</b> - Hemoglobin levels indicate significant anemia risk. Monitor closely and consider intervention.")
    elif prediction.anemia_risk == 'Medium':
        interpretations.append("• <b>Anemia Risk: MODERATE</b> - Some risk of anemia. Regular monitoring recommended.")
    else:
        interpretations.append("• <b>Anemia Risk: LOW</b> - Hemoglobin levels are within acceptable range.")
    
    if prediction.neutropenia_risk == 'High':
        interpretations.append("• <b>Neutropenia Risk: HIGH</b> - Low neutrophil count increases infection risk. Consider protective measures.")
    elif prediction.neutropenia_risk == 'Medium':
        interpretations.append("• <b>Neutropenia Risk: MODERATE</b> - Moderate risk of neutropenia. Monitor for signs of infection.")
    else:
        interpretations.append("• <b>Neutropenia Risk: LOW</b> - Neutrophil count is adequate.")
    
    if prediction.thrombocytopenia_risk == 'High':
        interpretations.append("• <b>Thrombocytopenia Risk: HIGH</b> - Low platelet count increases bleeding risk. Exercise caution.")
    elif prediction.thrombocytopenia_risk == 'Medium':
        interpretations.append("• <b>Thrombocytopenia Risk: MODERATE</b> - Moderate risk of thrombocytopenia. Monitor for bleeding.")
    else:
        interpretations.append("• <b>Thrombocytopenia Risk: LOW</b> - Platelet count is within normal range.")
    
    return "<br/>".join(interpretations)


def generate_qr_code(prediction_id: int, base_url: str) -> BytesIO:
    """Generate QR code for sharing prediction."""
    try:
        url = f"{base_url}/share/prediction/{prediction_id}"
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(url)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return buffer
    except Exception as e:
        raise Exception(f"Error generating QR code: {str(e)}")

