"""
QFLARE System Architecture - Mermaid Diagrams
Clean, structured diagrams in Mermaid format for documentation and presentations
"""

# QFLARE Overall System Block Diagram
overall_system_mermaid = """
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
"""

# Data Acquisition and Preparation Flow
data_acquisition_mermaid = """
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
"""

# Combined Dataset Creation
combined_dataset_mermaid = """
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
"""

# Federated Learning Flow
federated_learning_mermaid = """
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
"""

# PQC Handshake Protocol (without response times)
pqc_handshake_mermaid = """
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
"""

# Secure Aggregation Process
secure_aggregation_mermaid = """
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
"""

# Edge Node Architecture
edge_node_mermaid = """
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
"""

# QFLARE System Architecture Overview
system_architecture_mermaid = """
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
"""

# Threat Model and Security Layers
threat_model_mermaid = """
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
"""

if __name__ == "__main__":
    print("QFLARE Mermaid Diagram Codes")
    print("=" * 50)
    print("\n1. Overall System Block Diagram:")
    print(overall_system_mermaid)
    print("\n2. Data Acquisition and Preparation:")
    print(data_acquisition_mermaid)
    print("\n3. Combined Dataset Creation:")
    print(combined_dataset_mermaid)
    print("\n4. Federated Learning Flow:")
    print(federated_learning_mermaid)
    print("\n5. PQC Handshake Protocol:")
    print(pqc_handshake_mermaid)
    print("\n6. Secure Aggregation Process:")
    print(secure_aggregation_mermaid)
    print("\n7. Edge Node Architecture:")
    print(edge_node_mermaid)
    print("\n8. System Architecture Overview:")
    print(system_architecture_mermaid)
    print("\n9. Threat Model and Security Layers:")
    print(threat_model_mermaid)