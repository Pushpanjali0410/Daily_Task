"""Chat history export to PDF utility."""

from pathlib import Path
from typing import List, Dict, Any
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def export_chat_to_pdf(
    messages: List[Dict[str, str]],
    output_file: str | Path = "chats/chat_history.pdf"
) -> None:
    if not isinstance(messages, list):
        raise TypeError(f"messages must be a list, got {type(messages).__name__}")
    
    if not messages:
        logger.warning("No messages to export")
        return
    
    # Validate message format
    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError(f"Each message must be a dict, got {type(msg).__name__}")
        if "role" not in msg or "content" not in msg:
            raise ValueError("Each message must have 'role' and 'content' keys")
    
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create PDF document
        pdf = SimpleDocTemplate(
            str(output_path),
            pagesize=(8.5 * inch, 11 * inch),
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Create custom styles
        user_style = ParagraphStyle(
            'UserMessage',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            textColor='#0066cc',
            spaceAfter=6,
        )
        
        assistant_style = ParagraphStyle(
            'AssistantMessage',
            parent=styles['Normal'],
            fontName='Helvetica',
            textColor='#333333',
            spaceAfter=6,
        )
        
        # Build content
        content = []
        
        # Add header
        title = Paragraph(
            f"<b>Chat History Export</b><br/><i>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>",
            styles['Heading1']
        )
        content.append(title)
        content.append(Spacer(1, 0.3 * inch))
        
        # Add messages
        for i, msg in enumerate(messages, 1):
            role = msg.get("role", "unknown").upper()
            text = msg.get("content", "")
            
            # Choose style based on role
            if role == "USER":
                style = user_style
                prefix = "👤 You"
            else:
                style = assistant_style
                prefix = "🤖 Assistant"
            
            # Add message
            message_text = f"<b>{prefix}:</b> {text}"
            content.append(Paragraph(message_text, style))
            content.append(Spacer(1, 0.1 * inch))
            
            # Add page break every 5 messages
            if i % 5 == 0:
                content.append(PageBreak())
        
        # Build PDF
        pdf.build(content)
        logger.info(f"Chat history exported successfully to {output_path}")
    
    except ImportError as e:
        logger.error(f"reportlab library not found: {e}")
        raise ImportError("Please install reportlab: pip install reportlab")
    except Exception as e:
        logger.error(f"Error exporting chat to PDF: {e}")
        raise IOError(f"Failed to export chat to PDF {output_file}: {str(e)}")
