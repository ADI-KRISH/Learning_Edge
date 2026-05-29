import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def create_educational_pdf(filename="data/7. Object Detection and Deep Learning.pdf"):
    # Ensure target directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 1. Page template setup
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    # 2. Styling definitions
    styles = getSampleStyleSheet()
    
    # Custom elegant styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1e3a8a'), # Deep Dark Blue
        alignment=0, # Left-aligned
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#4b5563'), # Muted Gray
        alignment=0,
        spaceAfter=25
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
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
        'Heading2_Custom',
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
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1f2937'), # Charcoal
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=20,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    caption_style = ParagraphStyle(
        'Caption_Custom',
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
    story.append(Paragraph("Object Detection in Computer Vision: A Comprehensive Guide", title_style))
    story.append(Paragraph("Offline AI Tutor Courseware - Topic Study & Prerequisites", subtitle_style))
    story.append(Spacer(1, 10))
    
    # --- Section 1 ---
    story.append(Paragraph("1. Introduction to Object Detection", h1_style))
    story.append(Paragraph(
        "Object detection is a core computer vision task that involves both <b>localizing</b> where objects are located in an image "
        "and <b>classifying</b> what category each object belongs to. While standard image classification answers 'what is in this image?', "
        "object detection answers 'what is where in this image?' simultaneously.",
        body_style
    ))
    story.append(Paragraph(
        "To understand object detection, we must distinguish between three related tasks:",
        body_style
    ))
    
    story.append(Paragraph("• <b>Image Classification:</b> Predicts a single class label representing the primary object in the image. (e.g., 'Dog')", bullet_style))
    story.append(Paragraph("• <b>Object Localization:</b> Predicts a class label and outputs a single bounding box identifying the coordinates of that main object.", bullet_style))
    story.append(Paragraph("• <b>Object Detection:</b> Detects, localizes, and classifies multiple objects across the entire image scene, returning bounding boxes and confidence scores for each.", bullet_style))
    story.append(Spacer(1, 10))
    
    # --- Section 2 ---
    story.append(Paragraph("2. Traditional Object Detection Approaches", h1_style))
    story.append(Paragraph(
        "Before deep learning revolutionized computer vision, traditional pipelines relied on manual, hand-engineered feature descriptors combined with classifiers. "
        "A typical classic pipeline involved:",
        body_style
    ))
    
    story.append(Paragraph("• <b>Sliding Window search:</b> A cropping window of varying sizes sweeps across the entire image to crop sub-regions, feeding each sub-region to a classifier. This is computationally expensive.", bullet_style))
    story.append(Paragraph("• <b>Haar Cascade Classifiers:</b> Used famously in the Viola-Jones face detector (2001). It uses integral images and AdaBoost selection to detect simple rectangular features rapidly.", bullet_style))
    story.append(Paragraph("• <b>HOG + SVM:</b> Histogram of Oriented Gradients (HOG) is extracted as a robust feature representation and passed into a Support Vector Machine (SVM) classifier. Widely used for pedestrian detection.", bullet_style))
    story.append(Spacer(1, 10))
    
    # --- Section 3 ---
    story.append(Paragraph("3. Deep Learning Architectures", h1_style))
    story.append(Paragraph(
        "Modern deep learning detectors are categorized into two paradigms: <b>Two-Stage Detectors</b> (highly accurate but slower) and <b>One-Stage Detectors</b> (very fast, ideal for real-time applications).",
        body_style
    ))
    
    story.append(Paragraph("3.1 Two-Stage Object Detectors", h2_style))
    story.append(Paragraph(
        "Two-stage detectors divide the detection process into a proposal generation stage followed by a classification/refinement stage:",
        body_style
    ))
    story.append(Paragraph("• <b>R-CNN (Region-based CNN):</b> Uses Selective Search to propose ~2000 region proposals, warps each region, feeds it to a CNN to extract features, and classifies them using SVMs. Slow because it runs CNN 2000 times per image.", bullet_style))
    story.append(Paragraph("• <b>Fast R-CNN:</b> Resolves R-CNN's bottleneck by running the CNN *once* on the entire image to extract a feature map, then uses Region of Interest (RoI) Pooling to extract features for each proposal directly from the feature map.", bullet_style))
    story.append(Paragraph("• <b>Faster R-CNN:</b> Replaces Selective Search with a trainable <b>Region Proposal Network (RPN)</b> that shares convolutional features with the detection network, making the entire pipeline end-to-end trainable and significantly faster.", bullet_style))
    
    story.append(Spacer(1, 5))
    story.append(Paragraph("3.2 One-Stage Object Detectors", h2_style))
    story.append(Paragraph(
        "One-stage detectors frame object detection as a single regression problem, directly mapping pixel values to bounding boxes and class probabilities in a single pass:",
        body_style
    ))
    story.append(Paragraph("• <b>YOLO (You Only Look Once):</b> Divides the image into an S x S grid. Each grid cell predicts bounding boxes and confidence scores directly. Very fast and processes frames in real-time.", bullet_style))
    story.append(Paragraph("• <b>SSD (Single Shot MultiBox Detector):</b> Direct detection using convolutional feature maps of varying resolutions, allowing it to naturally handle objects of different scales without losing accuracy.", bullet_style))
    story.append(Spacer(1, 10))
    
    # --- Section 4 ---
    story.append(Paragraph("4. Key Evaluation Metrics", h1_style))
    story.append(Paragraph(
        "Object detection quality is evaluated using standard metrics to assess box accuracy and classification performance:",
        body_style
    ))
    
    # Summary Table of Metrics
    table_data = [
        [Paragraph("<b>Metric</b>", body_style), Paragraph("<b>Definition</b>", body_style), Paragraph("<b>Purpose</b>", body_style)],
        [
            Paragraph("<b>IoU</b><br/>(Intersection over Union)", body_style),
            Paragraph("Area of Overlap divided by Area of Union between the predicted box and ground truth box.", body_style),
            Paragraph("Measures how accurately the predicted bounding box aligns with the true object boundaries.", body_style)
        ],
        [
            Paragraph("<b>mAP</b><br/>(Mean Average Precision)", body_style),
            Paragraph("The average of the Average Precision (AP) calculated across all object categories.", body_style),
            Paragraph("Summarizes overall classification and localization performance into a single score.", body_style)
        ],
        [
            Paragraph("<b>NMS</b><br/>(Non-Maximum Suppression)", body_style),
            Paragraph("Post-processing algorithm that filters out overlapping candidate boxes for the same object.", body_style),
            Paragraph("Eliminates duplicate, highly-overlapping redundant boxes, keeping only the highest-scoring detection.", body_style)
        ]
    ]
    
    t = Table(table_data, colWidths=[1.5*inch, 2.5*inch, 2.5*inch])
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
    story.append(Paragraph("Table 1.1: Core metrics used in object detection systems.", caption_style))
    story.append(Spacer(1, 10))
    
    # --- Summary Section ---
    story.append(Paragraph("5. Summary & Key Takeaways", h1_style))
    story.append(Paragraph(
        "Choosing between one-stage and two-stage detectors depends heavily on the production hardware constraints. "
        "For real-time deployment on standard edge devices or mobile environments, one-stage models like <b>YOLOv8</b> or SSD are preferred "
        "due to their high frame-rate capabilities. For complex datasets requiring ultra-high localization accuracy (e.g., medical imaging), "
        "two-stage systems like Faster R-CNN or Cascade R-CNN remain standard.",
        body_style
    ))
    
    # Build the document
    doc.build(story)
    print(f"Successfully generated clean educational PDF at: {filename}")

def create_hadoop_pdf(filename="data/8. Hadoop and Distributed Systems.pdf"):
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
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'HadoopDocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1e3a8a'), # Deep Dark Blue
        alignment=0,
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'HadoopDocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#4b5563'),
        alignment=0,
        spaceAfter=25
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#0f766e'),
        spaceBefore=18,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom2',
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
        'Body_Custom2',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1f2937'),
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'Bullet_Custom2',
        parent=body_style,
        leftIndent=20,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    caption_style = ParagraphStyle(
        'Caption_Custom2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor('#4b5563'),
        alignment=1,
        spaceBefore=6,
        spaceAfter=12
    )

    story = []
    
    # --- Title Section ---
    story.append(Paragraph("Hadoop and Distributed Systems: HDFS & MapReduce Architectures", title_style))
    story.append(Paragraph("Offline AI Tutor Courseware - Distributed Computing Foundations", subtitle_style))
    story.append(Spacer(1, 10))
    
    # --- Section 1 ---
    story.append(Paragraph("1. Introduction to Distributed Systems and Big Data", h1_style))
    story.append(Paragraph(
        "A distributed system consists of multiple autonomous computers that communicate through a computer network and coordinate "
        "their actions by passing messages. Distributed systems are essential for processing 'Big Data' - datasets so large and complex "
        "that traditional data processing software is inadequate. Hadoop is the open-source industry standard framework developed to address this challenge.",
        body_style
    ))
    story.append(Paragraph(
        "Hadoop is based on a simple, yet powerful philosophy: <i>bring the computation to the data</i> instead of bringing the massive data "
        "to a single computing node. This eliminates the bandwidth bottleneck of network transfers.",
        body_style
    ))
    
    # --- Section 2 ---
    story.append(Paragraph("2. HDFS: Hadoop Distributed File System", h1_style))
    story.append(Paragraph(
        "HDFS is the primary storage system of Hadoop. It is designed to run on commodity hardware clusters, providing high-throughput "
        "access to application data and robust fault tolerance through data replication.",
        body_style
    ))
    story.append(Paragraph(
        "HDFS uses a master/slave architecture consisting of the following key nodes:",
        body_style
    ))
    story.append(Paragraph("• <b>NameNode (Master):</b> Manages the file system namespace, directories, and files. It tracks which DataNodes hold the blocks for a given file. It is a single point of failure in standard Hadoop 1.x configurations.", bullet_style))
    story.append(Paragraph("• <b>DataNode (Slave):</b> Stores and retrieves blocks of data as directed by the NameNode. DataNodes periodically send heartbeats and block reports to the NameNode to confirm their health.", bullet_style))
    story.append(Paragraph("• <b>Block Replication:</b> Files are split into large blocks (typically 128 MB or 256 MB) and distributed across the cluster. Each block is replicated (default is 3x) across different racks to survive node and rack failures.", bullet_style))
    story.append(Spacer(1, 10))
    
    # --- Section 3 ---
    story.append(Paragraph("3. MapReduce: The Distributed Computing Model", h1_style))
    story.append(Paragraph(
        "MapReduce is a programming model and software framework for writing applications that rapidly process vast amounts of data "
        "in parallel on large clusters of commodity hardware. It simplifies distributed programming by hiding the complexities of serialization, network transfer, and synchronization.",
        body_style
    ))
    story.append(Paragraph(
        "A MapReduce job executes in three main logical phases:",
        body_style
    ))
    story.append(Paragraph("1. <b>Map Phase:</b> Takes input key-value pairs, processes them, and emits intermediate key-value pairs. (e.g., parsing log files to count occurrences of specific status codes).", bullet_style))
    story.append(Paragraph("2. <b>Shuffle and Sort:</b> A system-level process that automatically groups intermediate keys and routes all values associated with the same intermediate key to the same Reducer node.", bullet_style))
    story.append(Paragraph("3. <b>Reduce Phase:</b> Aggregates or summarizes the values for each key, emitting the final output pairs to HDFS. (e.g., summing up counts for a specific status code).", bullet_style))
    story.append(Spacer(1, 10))
    
    # --- Section 4 ---
    story.append(Paragraph("4. Key Components of the Hadoop Ecosystem", h1_style))
    story.append(Paragraph(
        "The Hadoop architecture relies on a suite of integrated open-source tools to coordinate resources, manage datasets, and facilitate database storage:",
        body_style
    ))
    
    table_data = [
        [Paragraph("<b>Component</b>", body_style), Paragraph("<b>Function / Role</b>", body_style), Paragraph("<b>Primary Advantage</b>", body_style)],
        [
            Paragraph("<b>HDFS</b>", body_style),
            Paragraph("Distributed Storage layer.", body_style),
            Paragraph("Provides fault tolerance via 3x replication and high read/write throughput.", body_style)
        ],
        [
            Paragraph("<b>MapReduce</b>", body_style),
            Paragraph("Distributed Processing layer.", body_style),
            Paragraph("Processes massive datasets in parallel across thousands of nodes.", body_style)
        ],
        [
            Paragraph("<b>YARN</b>", body_style),
            Paragraph("Resource Manager & Job Scheduler.", body_style),
            Paragraph("Coordinates cluster CPU, RAM, and storage allocation among running jobs.", body_style)
        ],
        [
            Paragraph("<b>Apache Hive</b>", body_style),
            Paragraph("Data Warehousing SQL Interface.", body_style),
            Paragraph("Allows developers to query HDFS data using standard SQL instead of writing raw MapReduce Java code.", body_style)
        ]
    ]
    
    t = Table(table_data, colWidths=[1.2*inch, 2.8*inch, 2.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f3f4f6')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d1d5db')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    
    story.append(t)
    story.append(Paragraph("Table 2.1: Key frameworks inside the Hadoop distributed ecosystem.", caption_style))
    story.append(Spacer(1, 10))
    
    # --- Summary Section ---
    story.append(Paragraph("5. Summary & Key Takeaways", h1_style))
    story.append(Paragraph(
        "Hadoop is a foundational framework for distributed processing and big data storage. HDFS ensures data reliability "
        "through physical replication, while MapReduce enables highly scalable parallel batch computations. Modern architectures "
        "often pair HDFS with Apache Spark for real-time in-memory streaming, but Hadoop's distributed design patterns remain the core of big data engineering.",
        body_style
    ))
    
    doc.build(story)
    print(f"Successfully generated clean Hadoop PDF at: {filename}")

if __name__ == "__main__":
    create_educational_pdf()
    create_hadoop_pdf()
