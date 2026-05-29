import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def create_rag_pdf(filename="data/9. RAG and Vectorless RAG.pdf"):
    print(f"Creating professional educational PDF at: {filename}...")
    # Ensure target directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Page template setup
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    # Custom elegant styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'RAGDocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1e3a8a'), # Deep Dark Blue
        alignment=0,
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'RAGDocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#4b5563'),
        alignment=0,
        spaceAfter=25
    )
    
    h1_style = ParagraphStyle(
        'RAGHeading1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#0f766e'), # Teal Accent
        spaceBefore=18,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'RAGHeading2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0f766e'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'RAGBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1f2937'), # Charcoal
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'RAGBullet',
        parent=body_style,
        leftIndent=20,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    caption_style = ParagraphStyle(
        'RAGCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor('#4b5563'),
        alignment=1, # Center
        spaceBefore=6,
        spaceAfter=12
    )

    story = []
    
    # --- Title Section ---
    story.append(Paragraph("Retrieval-Augmented Generation & Vectorless RAG Architectures", title_style))
    story.append(Paragraph("Offline AI Tutor Courseware - Advanced Natural Language Processing", subtitle_style))
    story.append(Spacer(1, 10))
    
    # --- Section 1 ---
    story.append(Paragraph("1. Introduction to Retrieval-Augmented Generation (RAG)", h1_style))
    story.append(Paragraph(
        "Standard Large Language Models (LLMs) are limited by their parametric knowledge—the facts they memorized "
        "during pretraining. Consequently, they struggle with proprietary data, real-time events, and are prone to "
        "hallucinating incorrect facts. <b>Retrieval-Augmented Generation (RAG)</b> is an architectural pattern that "
        "resolves these issues by supplementing LLM inputs with relevant contextual facts retrieved from external databases.",
        body_style
    ))
    story.append(Paragraph(
        "A typical vector-based RAG pipeline involves three distinct steps:",
        body_style
    ))
    
    story.append(Paragraph("• <b>Ingestion:</b> Documents are divided into small, cohesive text chunks. Each chunk is passed through an embedding model to generate a high-dimensional vector representation, which is stored in a specialized Vector Database (e.g., ChromaDB, Pinecone).", bullet_style))
    story.append(Paragraph("• <b>Retrieval:</b> When a user asks a question, the question is embedded using the same model. The vector database performs a cosine similarity or Euclidean distance search to identify and return the top-K chunks most semantically similar to the query.", bullet_style))
    story.append(Paragraph("• <b>Generation:</b> The retrieved context chunks are joined alongside the original user question inside a specialized prompt template, which is sent to the LLM to formulate an accurate, evidence-backed answer.", bullet_style))
    story.append(Spacer(1, 10))
    
    # --- Section 2 ---
    story.append(Paragraph("2. The Limitations of Traditional Vector-Based RAG", h1_style))
    story.append(Paragraph(
        "While vector databases are excellent for finding general semantic similarities, they suffer from several fundamental limitations:",
        body_style
    ))
    
    story.append(Paragraph("• <b>Loss of Fine-Grained Lexical Match:</b> Semantic embeddings map 'words of similar meaning' together, but they often fail at matching exact codes, IDs, or highly specific jargon (e.g., part numbers or custom function names).", bullet_style))
    story.append(Paragraph("• <b>Lack of Relational Reasoning:</b> Embedding vectors represent isolated chunks of text. They cannot trace linkages or bridge concepts across multiple documents (e.g., answering 'which libraries depend on version 2.4?' when the dependency tree is scattered across 10 files).", bullet_style))
    story.append(Paragraph("• <b>The Out-of-Context Chunking Problem:</b> Breaking documents into rigid, sliding-window chunks (e.g., 256 tokens) strips away critical surrounding context, table headers, or document meta-structures, leading to fragmented and incomplete answers.", bullet_style))
    story.append(Spacer(1, 10))
    
    # --- Section 3 ---
    story.append(Paragraph("3. Vectorless RAG paradigms", h1_style))
    story.append(Paragraph(
        "To address vector constraints, <b>Vectorless RAG</b> patterns retrieve contextual facts without using high-dimensional dense embeddings or vector spaces. These methods rely on alternative data structures and classical query mechanisms:",
        body_style
    ))
    
    story.append(Paragraph("3.1 Keyword and Lexical Retrieval (BM25)", h2_style))
    story.append(Paragraph(
        "Lexical vectorless search relies on classical term frequency and inverse document frequency algorithms. "
        "<b>BM25 (Best Match 25)</b> is the gold standard for keyword matching. It scores candidate chunks by calculating the "
        "frequency of exact search terms in a chunk relative to how commonly the terms appear across all documents. "
        "BM25 is highly effective at resolving specific technical queries, exact names, or numerical symbols where semantic synonyms are irrelevant.",
        body_style
    ))
    
    story.append(Paragraph("3.2 Graph RAG (Knowledge Graphs)", h2_style))
    story.append(Paragraph(
        "<b>Graph RAG</b> represents educational materials as a structured <b>Knowledge Graph</b> (KG), where nodes represent "
        "key concepts (e.g., 'K-Means', 'Unsupervised Learning') and edges represent relationships (e.g., 'K-Means' <i>is a type of</i> 'Unsupervised Learning'). "
        "Instead of searching vector spaces, Graph RAG extracts entities from the user's question, navigates the graph's connections to find "
        "prerequisites or dependent concepts, and injects this deep relational map into the prompt. This provides unparalleled structural understanding.",
        body_style
    ))
    
    story.append(Paragraph("3.3 Relational SQL RAG", h2_style))
    story.append(Paragraph(
        "For tabular, relational data (e.g., structured databases or spreadsheets), vector matching performs poorly. "
        "<b>SQL RAG</b> uses the LLM to dynamically generate SQL query statements based on the schema definitions and the user's intent. "
        "The system executes the SQL query against a relational database and returns raw structured tables to the LLM to perform numerical calculations or aggregate results.",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    # --- Section 4 ---
    story.append(Paragraph("4. Architectural Comparison", h1_style))
    story.append(Paragraph(
        "Modern enterprise-grade systems rarely use vector search in isolation. Instead, they combine vector and vectorless systems using a hybrid architecture:",
        body_style
    ))
    
    # Summary Table of Architectures
    table_data = [
        [Paragraph("<b>Approach</b>", body_style), Paragraph("<b>Retrieval Mechanism</b>", body_style), Paragraph("<b>Strengths</b>", body_style), Paragraph("<b>Weaknesses</b>", body_style)],
        [
            Paragraph("<b>Vector RAG</b>", body_style),
            Paragraph("Cosine similarity on dense neural embeddings.", body_style),
            Paragraph("Finds conceptual synonyms; highly robust to grammatical variance.", body_style),
            Paragraph("Fails at exact keyword matching; cannot query deep relational connections.", body_style)
        ],
        [
            Paragraph("<b>Lexical RAG</b><br/>(BM25)", body_style),
            Paragraph("Term frequency-inverse document frequency scoring.", body_style),
            Paragraph("Excellent for serial codes, exact IDs, specific names, and command syntaxes.", body_style),
            Paragraph("Cannot resolve synonyms (e.g., will miss 'car' if query uses 'vehicle').", body_style)
        ],
        [
            Paragraph("<b>Graph RAG</b>", body_style),
            Paragraph("Structured entity linkages and graph traversal.", body_style),
            Paragraph("Captures dependencies, hierarchical schemas, and cross-document reasoning.", body_style),
            Paragraph("Requires complex entity extraction; higher computational indexing cost.", body_style)
        ]
    ]
    
    t = Table(table_data, colWidths=[1.1*inch, 1.8*inch, 1.8*inch, 1.8*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f3f4f6')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d1d5db')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    
    story.append(t)
    story.append(Paragraph("Table 3.1: Technical comparison of Vector and Vectorless RAG paradigms.", caption_style))
    story.append(Spacer(1, 10))
    
    # --- Summary Section ---
    story.append(Paragraph("5. Summary & Key Takeaways", h1_style))
    story.append(Paragraph(
        "Optimal retrieval-augmented generation relies on <b>Hybrid Search</b>—combining the strengths of dense vector "
        "similarity (for conceptual understanding) and sparse lexical BM25 matching (for precise terminology) in tandem "
        "with reciprocal rank fusion (RRF). By anchoring LLMs to both vector-based and vectorless paradigms, systems can "
        "provide highly reliable, contextually comprehensive, and audit-verifiable educational tutoring.",
        body_style
    ))
    
    # Build the document
    doc.build(story)
    print(f"Successfully generated clean educational RAG PDF at: {filename}")

if __name__ == "__main__":
    # 1. Generate PDF
    create_rag_pdf()
    
    # 2. Trigger Ingestion to ChromaDB & Knowledge Graph
    print("\nStarting automatic ingestion to vector DB...")
    from app.rag.ingestion import DocumentIngestionPipeline
    pipeline = DocumentIngestionPipeline()
    pipeline.ingest_documents()
    print("\nAll done! The new RAG & Vectorless RAG topic is fully indexed and ready to study!")
