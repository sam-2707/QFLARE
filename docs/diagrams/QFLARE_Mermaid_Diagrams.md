# QFLARE System Architecture - Mermaid Diagrams

This document contains all QFLARE system diagrams in Mermaid format for easy integration into documentation, presentations, and markdown files.

## 1. Overall System Block Diagram

```mermaid
flowchart TD
    A[Dataset Acquisition and Preparation] --> B[Dataset Labeling and Classification]
    B --> C[Model Training and Evaluation]
    C --> D[Real-time Prediction Pipeline]
    D --> E[NIDS Dashboard]

    style A fill:#f9f9f9,stroke:#333,stroke-width:2px
    style B fill:#f9f9f9,stroke:#333,stroke-width:2px
    style C fill:#f9f9f9,stroke:#333,stroke-width:2px
    style D fill:#f9f9f9,stroke:#333,stroke-width:2px
    style E fill:#f9f9f9,stroke:#333,stroke-width:2px
```

## 2. Data Acquisition and Preparation

```mermaid
flowchart TD
    A[Bot-IoT Dataset<br/>5% dataset, 4x files] --> B[Analysis<br/>class count, attack types, features]
    B --> C[Feature Selection]
    C --> D[Data Extraction<br/>merging, shuffling]
    D --> E[Export as DB1.csv]

    style A fill:#f9f9f9,stroke:#333,stroke-width:2px
    style B fill:#f9f9f9,stroke:#333,stroke-width:2px
    style C fill:#f9f9f9,stroke:#333,stroke-width:2px
    style D fill:#f9f9f9,stroke:#333,stroke-width:2px
    style E fill:#f9f9f9,stroke:#333,stroke-width:2px
```

## 3. Combined Dataset Creation

```mermaid
flowchart TD
    A[Bot-IoT 5% Dataset] --> B[File 1]
    A --> C[File 2]
    A --> D[File 3]
    A --> E[File 4]
    
    B --> F[Merging and Shuffling]
    C --> F
    D --> F
    E --> F
    
    F --> G[DB1.csv<br/>Combined Dataset]

    style A fill:#f9f9f9,stroke:#333,stroke-width:2px
    style B fill:#f9f9f9,stroke:#333,stroke-width:2px
    style C fill:#f9f9f9,stroke:#333,stroke-width:2px
    style D fill:#f9f9f9,stroke:#333,stroke-width:2px
    style E fill:#f9f9f9,stroke:#333,stroke-width:2px
    style F fill:#f9f9f9,stroke:#333,stroke-width:2px
    style G fill:#f9f9f9,stroke:#333,stroke-width:2px
```

## 4. QFLARE Federated Learning Flow

```mermaid
flowchart TD
    A[Edge Node 1<br/>Local Training] <--> B[Central Server<br/>Global Model]
    C[Edge Node 2<br/>Local Training] <--> B
    D[Edge Node 3<br/>Local Training] <--> B
    
    B --> E[Secure Aggregation<br/>PQC Protected]

    style A fill:#e8f4fd,stroke:#333,stroke-width:2px
    style B fill:#e8f8e8,stroke:#333,stroke-width:2px
    style C fill:#e8f4fd,stroke:#333,stroke-width:2px
    style D fill:#e8f4fd,stroke:#333,stroke-width:2px
    style E fill:#ffe8f8,stroke:#333,stroke-width:2px
```

## 5. PQC Handshake Protocol (No Response Times)

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    
    Note over C: 1. Generate Kyber Keypair<br/>(Private/Public Keys)
    C->>S: 2. Send Public Key + Connection Request
    Note over S: 3. Verify Client Key<br/>Generate Server Keypair
    S->>C: 4. Send Server Public Key + Encrypted Session Key
    Note over C: 5. Decrypt Session Key<br/>Verify Server Identity
    Note over C: 6. Generate Dilithium<br/>Signature for Auth
    C->>S: 7. Send Signed Authentication
    Note over S: 8. Verify Signature<br/>Establish Secure Channel
    Note over C,S: 9. Secure Communication Channel Active
```

## 6. Secure Aggregation Process

```mermaid
flowchart TD
    A[Local Model Updates<br/>Edge Nodes] --> B[Apply Differential Privacy<br/>Noise Addition]
    B --> C[Encrypt with PQC<br/>CRYSTALS-Kyber]
    C --> D[Secure Multi-Party<br/>Computation]
    D --> E[Aggregate Encrypted<br/>Updates]
    E --> F[Global Model Update<br/>Central Server]

    style A fill:#e8f4fd,stroke:#333,stroke-width:2px
    style B fill:#ffe8f8,stroke:#333,stroke-width:2px
    style C fill:#ffe8f8,stroke:#333,stroke-width:2px
    style D fill:#ffe8f8,stroke:#333,stroke-width:2px
    style E fill:#ffe8f8,stroke:#333,stroke-width:2px
    style F fill:#e8f8e8,stroke:#333,stroke-width:2px
```

## 7. Edge Node Architecture

```mermaid
flowchart TD
    A[Local Data Storage<br/>Encrypted] --> B[Data Preprocessing<br/>Feature Engineering]
    B --> C[Local Model Training<br/>Privacy-Preserving]
    C --> D[Gradient Computation<br/>Differential Privacy]
    D --> E[PQC Encryption<br/>Model Updates]
    E --> F[Secure Communication<br/>To Central Server]

    style A fill:#f0f0f0,stroke:#333,stroke-width:2px
    style B fill:#f0f0f0,stroke:#333,stroke-width:2px
    style C fill:#e8f4fd,stroke:#333,stroke-width:2px
    style D fill:#ffe8f8,stroke:#333,stroke-width:2px
    style E fill:#ffe8f8,stroke:#333,stroke-width:2px
    style F fill:#e8e8ff,stroke:#333,stroke-width:2px
```

## 8. QFLARE System Architecture Overview

```mermaid
flowchart TB
    subgraph "Client Layer"
        A[Mobile Devices]
        B[IoT Sensors]
        C[Edge Compute]
        D[Desktop Clients]
    end
    
    subgraph "Edge Layer"
        E[Edge Node 1]
        F[Edge Node 2]
        G[Edge Node 3]
    end
    
    subgraph "Secure Communication"
        H[PQC Encryption<br/>CRYSTALS-Kyber]
        I[Digital Signatures<br/>CRYSTALS-Dilithium]
    end
    
    subgraph "Central Server"
        J[Global Model]
        K[Secure Aggregation]
        L[Model Distribution]
    end
    
    subgraph "Privacy Protection"
        M[Differential Privacy]
        N[Homomorphic Encryption]
        O[Secure Multi-Party Computation]
    end
    
    A --> E
    B --> F
    C --> G
    D --> E
    
    E <--> H
    F <--> H
    G <--> H
    
    H <--> I
    I <--> J
    
    J <--> K
    K <--> L
    
    E -.-> M
    F -.-> N
    G -.-> O
    
    M --> K
    N --> K
    O --> K

    style A fill:#e8f4fd,stroke:#333,stroke-width:2px
    style B fill:#e8f4fd,stroke:#333,stroke-width:2px
    style C fill:#e8f4fd,stroke:#333,stroke-width:2px
    style D fill:#e8f4fd,stroke:#333,stroke-width:2px
    style E fill:#ffe8cc,stroke:#333,stroke-width:2px
    style F fill:#ffe8cc,stroke:#333,stroke-width:2px
    style G fill:#ffe8cc,stroke:#333,stroke-width:2px
    style H fill:#ffe8f8,stroke:#333,stroke-width:2px
    style I fill:#ffe8f8,stroke:#333,stroke-width:2px
    style J fill:#e8f8e8,stroke:#333,stroke-width:2px
    style K fill:#e8f8e8,stroke:#333,stroke-width:2px
    style L fill:#e8f8e8,stroke:#333,stroke-width:2px
    style M fill:#ffe8e8,stroke:#333,stroke-width:2px
    style N fill:#ffe8e8,stroke:#333,stroke-width:2px
    style O fill:#ffe8e8,stroke:#333,stroke-width:2px
```

## 9. Threat Model and Security Layers

```mermaid
flowchart TB
    subgraph "Attack Vectors"
        A[Data Poisoning]
        B[Model Inversion]
        C[Gradient Leakage]
        D[Quantum Attacks]
        E[Byzantine Attacks]
    end
    
    subgraph "QFLARE Protection Layers"
        F[Post-Quantum Cryptography]
        G[Differential Privacy]
        H[Secure Aggregation]
        I[Byzantine Fault Tolerance]
        J[Homomorphic Encryption]
    end
    
    subgraph "Security Validation"
        K[Continuous Monitoring]
        L[Anomaly Detection]
        M[Performance Metrics]
        N[Compliance Auditing]
    end
    
    A -.-> G
    B -.-> J
    C -.-> H
    D -.-> F
    E -.-> I
    
    F --> K
    G --> L
    H --> M
    I --> N
    J --> K

    style A fill:#ffe8e8,stroke:#d33,stroke-width:2px
    style B fill:#ffe8e8,stroke:#d33,stroke-width:2px
    style C fill:#ffe8e8,stroke:#d33,stroke-width:2px
    style D fill:#ffe8e8,stroke:#d33,stroke-width:2px
    style E fill:#ffe8e8,stroke:#d33,stroke-width:2px
    style F fill:#e8f8e8,stroke:#3d8,stroke-width:2px
    style G fill:#e8f8e8,stroke:#3d8,stroke-width:2px
    style H fill:#e8f8e8,stroke:#3d8,stroke-width:2px
    style I fill:#e8f8e8,stroke:#3d8,stroke-width:2px
    style J fill:#e8f8e8,stroke:#3d8,stroke-width:2px
    style K fill:#e8f4fd,stroke:#38d,stroke-width:2px
    style L fill:#e8f4fd,stroke:#38d,stroke-width:2px
    style M fill:#e8f4fd,stroke:#38d,stroke-width:2px
    style N fill:#e8f4fd,stroke:#38d,stroke-width:2px
```

## Usage Instructions

### For GitHub/GitLab/Documentation
- Copy any diagram code block and paste it into your markdown files
- The diagrams will render automatically in most modern markdown viewers
- GitHub, GitLab, and most documentation platforms support Mermaid natively

### For Presentations
- Use Mermaid Live Editor (https://mermaid.live) to generate PNG/SVG files
- Copy the code, paste into the editor, and download the rendered image
- Perfect for PowerPoint, Google Slides, or any presentation software

### For Web Integration
- Include Mermaid.js library in your web pages
- Paste the diagram code directly into your HTML/markdown content
- Diagrams will render client-side with full interactivity

### Color Coding
- **Light Blue (#e8f4fd)**: Client/Edge components
- **Light Orange (#ffe8cc)**: Edge processing nodes  
- **Light Green (#e8f8e8)**: Central server components
- **Light Pink (#ffe8f8)**: Cryptographic/Security components
- **Light Gray (#f0f0f0)**: Data storage/processing
- **Light Red (#ffe8e8)**: Security threats/monitoring

All diagrams are designed for professional documentation and presentations with clean, consistent styling.