from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)


def export_chat(messages):

    pdf = SimpleDocTemplate(
        "chat_history.pdf"
    )

    styles = getSampleStyleSheet()

    content = []

    for msg in messages:

        content.append(
            Paragraph(
                str(msg),
                styles["Normal"]
            )
        )

    pdf.build(content)