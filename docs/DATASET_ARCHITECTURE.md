# QFLARE Dataset Architecture - Mermaid Diagrams

## 1. Overall Dataset Flow Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        MNIST[MNIST Dataset<br/>60k train + 10k test]
        CIFAR[CIFAR-10<br/>50k train + 10k test]
        Custom[Custom Datasets<br/>User uploaded]
    end
    
    subgraph "Data Preprocessing"
        Load[Data Loader]
        Split[Data Splitter]
        Norm[Normalization]
        Aug[Augmentation]
        
        MNIST --> Load
        CIFAR --> Load
        Custom --> Load
        Load --> Split
        Split --> Norm
        Norm --> Aug
    end
    
    subgraph "Federated Distribution"
        Coord[Coordinator]
        
        Aug --> Coord
        
        Coord --> Node1[Edge Node 1<br/>IID Partition]
        Coord --> Node2[Edge Node 2<br/>IID Partition]
        Coord --> Node3[Edge Node 3<br/>Non-IID Partition]
        Coord --> NodeN[Edge Node N<br/>Skewed Partition]
    end
    
    subgraph "Local Training"
        Node1 --> Train1[Local Model 1]
        Node2 --> Train2[Local Model 2]
        Node3 --> Train3[Local Model 3]
        NodeN --> TrainN[Local Model N]
    end
    
    subgraph "Privacy Layer"
        Train1 --> DP1[Differential Privacy<br/>ε-δ Mechanism]
        Train2 --> DP2[Differential Privacy<br/>ε-δ Mechanism]
        Train3 --> DP3[Differential Privacy<br/>ε-δ Mechanism]
        TrainN --> DPN[Differential Privacy<br/>ε-δ Mechanism]
    end
    
    subgraph "Aggregation"
        DP1 --> Agg[Secure Aggregator<br/>Byzantine Detection]
        DP2 --> Agg
        DP3 --> Agg
        DPN --> Agg
        
        Agg --> Global[Global Model]
    end
    
    Global -.Next Round.-> Coord
    
    style MNIST fill:#e1f5ff
    style CIFAR fill:#e1f5ff
    style Custom fill:#e1f5ff
    style DP1 fill:#fff3e0
    style DP2 fill:#fff3e0
    style DP3 fill:#fff3e0
    style DPN fill:#fff3e0
    style Agg fill:#ffebee
    style Global fill:#e8f5e9
```

## 2. Data Partitioning Strategies

```mermaid
graph LR
    subgraph "Dataset"
        D[Original Dataset<br/>D = {x₁, x₂, ..., xₙ}]
    end
    
    subgraph "IID Partitioning"
        D --> IID_Split[Random Shuffle<br/>& Equal Split]
        IID_Split --> IID1[Partition 1<br/>Uniform Distribution]
        IID_Split --> IID2[Partition 2<br/>Uniform Distribution]
        IID_Split --> IID3[Partition 3<br/>Uniform Distribution]
    end
    
    subgraph "Non-IID Partitioning"
        D --> NIID_Split[Class-based Split<br/>Dirichlet α=0.5]
        NIID_Split --> NIID1[Partition 1<br/>Classes: 0,1,2]
        NIID_Split --> NIID2[Partition 2<br/>Classes: 3,4,5]
        NIID_Split --> NIID3[Partition 3<br/>Classes: 6,7,8,9]
    end
    
    subgraph "Quantity Skewed"
        D --> QS_Split[Unequal Split<br/>Power Law]
        QS_Split --> QS1[Partition 1<br/>50% data]
        QS_Split --> QS2[Partition 2<br/>30% data]
        QS_Split --> QS3[Partition 3<br/>20% data]
    end
    
    style D fill:#e3f2fd
    style IID1 fill:#c8e6c9
    style IID2 fill:#c8e6c9
    style IID3 fill:#c8e6c9
    style NIID1 fill:#ffccbc
    style NIID2 fill:#ffccbc
    style NIID3 fill:#ffccbc
    style QS1 fill:#f0f4c3
    style QS2 fill:#f0f4c3
    style QS3 fill:#f0f4c3
```

## 3. Dataset Storage Structure

```mermaid
graph TB
    subgraph "Storage Hierarchy"
        Root[/data/]
        
        Root --> MNIST_Dir[MNIST/]
        Root --> Models[models/]
        Root --> Updates[updates/]
        Root --> Keys[keys/]
        Root --> Logs[logs/]
        
        MNIST_Dir --> Raw[raw/<br/>Original files]
        MNIST_Dir --> Processed[processed/<br/>train.pt, test.pt]
        MNIST_Dir --> Partitions[partitions/<br/>node_1.pt...node_n.pt]
        
        Models --> Checkpoints[checkpoints/<br/>epoch_*.pth]
        Models --> Global[global/<br/>global_model.pth]
        Models --> Local[local/<br/>node_*_model.pth]
        
        Updates --> Round[round_*/<br/>Gradient updates]
        Updates --> Encrypted[encrypted/<br/>Secure updates]
        
        Keys --> Public[public/<br/>Public keys]
        Keys --> Private[private/<br/>Private keys]
        Keys --> Quantum[quantum/<br/>Post-quantum keys]
        
        Logs --> Training[training/<br/>metrics.json]
        Logs --> Security[security/<br/>audit.log]
        Logs --> Privacy[privacy/<br/>epsilon_budget.json]
    end
    
    style Root fill:#e1f5ff
    style MNIST_Dir fill:#fff3e0
    style Models fill:#f3e5f5
    style Updates fill:#e8f5e9
    style Keys fill:#ffebee
    style Logs fill:#fce4ec
```

## 4. Data Flow During Federated Training

```mermaid
sequenceDiagram
    participant C as Coordinator
    participant N1 as Edge Node 1
    participant N2 as Edge Node 2
    participant N3 as Edge Node 3
    participant A as Aggregator
    
    Note over C: Round Start
    C->>C: Load Global Model
    
    par Parallel Data Distribution
        C->>N1: Send Model + Config
        C->>N2: Send Model + Config
        C->>N3: Send Model + Config
    end
    
    Note over N1,N3: Local Training
    
    par Local Training
        N1->>N1: Load Local Dataset<br/>(IID Partition)
        N2->>N2: Load Local Dataset<br/>(IID Partition)
        N3->>N3: Load Local Dataset<br/>(Non-IID Partition)
    end
    
    par Compute Gradients
        N1->>N1: Forward Pass<br/>Backward Pass
        N2->>N2: Forward Pass<br/>Backward Pass
        N3->>N3: Forward Pass<br/>Backward Pass
    end
    
    Note over N1,N3: Apply Privacy
    
    par Apply Differential Privacy
        N1->>N1: Add Gaussian Noise<br/>ε=0.1, δ=1e-6
        N2->>N2: Add Gaussian Noise<br/>ε=0.1, δ=1e-6
        N3->>N3: Add Gaussian Noise<br/>ε=0.1, δ=1e-6
    end
    
    par Upload Updates
        N1->>A: Encrypted Update 1
        N2->>A: Encrypted Update 2
        N3->>A: Encrypted Update 3
    end
    
    A->>A: Byzantine Detection<br/>Remove Outliers
    A->>A: Secure Aggregation<br/>Weighted Average
    
    A->>C: New Global Model
    C->>C: Update Global Model
    
    Note over C: Round Complete
```

## 5. Dataset Class Structure

```mermaid
classDiagram
    class Dataset {
        +String name
        +int num_samples
        +List~Tensor~ data
        +List~int~ labels
        +load()
        +preprocess()
        +get_batch()
    }
    
    class MNISTDataset {
        +int image_size
        +int num_classes
        +download()
        +transform()
    }
    
    class CIFAR10Dataset {
        +int image_size
        +int num_classes
        +download()
        +augment()
    }
    
    class CustomDataset {
        +String data_path
        +Dict schema
        +validate()
        +load_from_file()
    }
    
    class FederatedDataLoader {
        +int num_nodes
        +String partition_type
        +float alpha
        +partition_data()
        +get_node_data()
        +balance_classes()
    }
    
    class PrivateDataset {
        +float epsilon
        +float delta
        +int clip_norm
        +add_noise()
        +compute_privacy_budget()
    }
    
    class DataAugmentor {
        +List~Transform~ transforms
        +apply_augmentation()
        +random_crop()
        +random_flip()
    }
    
    Dataset <|-- MNISTDataset
    Dataset <|-- CIFAR10Dataset
    Dataset <|-- CustomDataset
    
    Dataset --> FederatedDataLoader
    FederatedDataLoader --> PrivateDataset
    Dataset --> DataAugmentor
```

## 6. Privacy Budget Flow

```mermaid
graph TD
    subgraph "Privacy Budget Management"
        Init[Initialize Budget<br/>ε_total = 1.0<br/>δ = 1e-6]
        
        Init --> Round1[Round 1]
        Round1 --> Consume1[Consume: ε₁ = 0.1]
        Consume1 --> Remain1[Remaining: 0.9]
        
        Remain1 --> Round2[Round 2]
        Round2 --> Consume2[Consume: ε₂ = 0.1]
        Consume2 --> Remain2[Remaining: 0.8]
        
        Remain2 --> Round3[Round 3]
        Round3 --> Consume3[Consume: ε₃ = 0.1]
        Consume3 --> Remain3[Remaining: 0.7]
        
        Remain3 --> RoundN[Round N]
        RoundN --> ConsumeN[Consume: εₙ = 0.1]
        ConsumeN --> RemainN[Remaining: ε_remain]
        
        RemainN --> Check{ε_remain > 0?}
        Check -->|Yes| Continue[Continue Training]
        Check -->|No| Stop[Stop Training<br/>Budget Exhausted]
        
        Continue --> RoundN
    end
    
    style Init fill:#e8f5e9
    style Stop fill:#ffebee
    style Continue fill:#e3f2fd
```

## 7. Dataset Metadata Schema

```mermaid
erDiagram
    DATASET ||--o{ PARTITION : contains
    DATASET ||--o{ METADATA : has
    PARTITION ||--o{ SAMPLE : contains
    METADATA ||--o{ PRIVACY_CONFIG : defines
    
    DATASET {
        string dataset_id PK
        string name
        string type
        int total_samples
        int num_classes
        timestamp created_at
    }
    
    PARTITION {
        string partition_id PK
        string dataset_id FK
        int node_id
        string partition_type
        int num_samples
        float alpha
    }
    
    SAMPLE {
        string sample_id PK
        string partition_id FK
        blob data
        int label
        float weight
    }
    
    METADATA {
        string metadata_id PK
        string dataset_id FK
        json preprocessing
        json augmentation
        json statistics
    }
    
    PRIVACY_CONFIG {
        string config_id PK
        string metadata_id FK
        float epsilon
        float delta
        int clip_norm
        string mechanism
    }
```

## 8. Byzantine Detection on Dataset

```mermaid
graph TB
    subgraph "Update Collection"
        U1[Update 1<br/>Node 1]
        U2[Update 2<br/>Node 2]
        U3[Update 3<br/>Node 3 - Byzantine]
        U4[Update 4<br/>Node 4]
        UN[Update N<br/>Node N]
    end
    
    subgraph "Statistical Analysis"
        U1 --> Stats[Compute Statistics]
        U2 --> Stats
        U3 --> Stats
        U4 --> Stats
        UN --> Stats
        
        Stats --> Mean[Mean: μ]
        Stats --> Std[Std Dev: σ]
        Stats --> Median[Median: m]
    end
    
    subgraph "Outlier Detection"
        Mean --> Z[Z-Score Test<br/>|z| > 3?]
        Std --> Z
        
        Z --> Check1{U1 Normal?}
        Z --> Check2{U2 Normal?}
        Z --> Check3{U3 Outlier?}
        Z --> Check4{U4 Normal?}
        Z --> CheckN{UN Normal?}
        
        Check1 -->|Yes| Accept1[✓ Accept]
        Check2 -->|Yes| Accept2[✓ Accept]
        Check3 -->|No| Reject3[✗ Reject Byzantine]
        Check4 -->|Yes| Accept4[✓ Accept]
        CheckN -->|Yes| AcceptN[✓ Accept]
    end
    
    subgraph "Aggregation"
        Accept1 --> Agg[Weighted Average]
        Accept2 --> Agg
        Accept4 --> Agg
        AcceptN --> Agg
        
        Agg --> Final[Final Model Update]
    end
    
    Reject3 --> Log[Security Log<br/>Byzantine Attack Detected]
    
    style U3 fill:#ffebee
    style Check3 fill:#ffccbc
    style Reject3 fill:#ef5350
    style Log fill:#ffcdd2
    style Accept1 fill:#c8e6c9
    style Accept2 fill:#c8e6c9
    style Accept4 fill:#c8e6c9
    style AcceptN fill:#c8e6c9
    style Final fill:#a5d6a7
```

## 9. Data Encryption Pipeline

```mermaid
graph LR
    subgraph "Plain Data"
        Plain[Local Dataset<br/>Unencrypted]
    end
    
    subgraph "Encryption Layers"
        Plain --> Layer1[Layer 1: AES-256<br/>Symmetric Encryption]
        Layer1 --> Layer2[Layer 2: RSA-4096<br/>Key Encryption]
        Layer2 --> Layer3[Layer 3: CRYSTALS-Kyber<br/>Post-Quantum]
    end
    
    subgraph "Secure Storage"
        Layer3 --> Store[Encrypted Storage<br/>Triple-Layer Protection]
    end
    
    subgraph "Transmission"
        Store --> TLS[TLS 1.3<br/>Transport Layer]
        TLS --> Network[Network<br/>Secure Channel]
    end
    
    subgraph "Decryption"
        Network --> D_Layer3[Decrypt: Kyber]
        D_Layer3 --> D_Layer2[Decrypt: RSA]
        D_Layer2 --> D_Layer1[Decrypt: AES]
        D_Layer1 --> Receiver[Receiver Node<br/>Plain Data]
    end
    
    style Plain fill:#e3f2fd
    style Layer1 fill:#fff3e0
    style Layer2 fill:#f3e5f5
    style Layer3 fill:#ffebee
    style Store fill:#e8f5e9
    style Receiver fill:#e3f2fd
```

## 10. Real-time Dataset Metrics Dashboard

```mermaid
graph TB
    subgraph "Data Sources"
        DS[Training Data]
        DS --> Metrics[Metrics Collector]
    end
    
    subgraph "Metrics Pipeline"
        Metrics --> Accuracy[Accuracy<br/>per Round]
        Metrics --> Loss[Loss<br/>per Round]
        Metrics --> Privacy[Privacy Budget<br/>ε consumed]
        Metrics --> Security[Byzantine<br/>Detections]
        Metrics --> Performance[Training Time<br/>per Round]
    end
    
    subgraph "Storage"
        Accuracy --> TS[Time Series DB<br/>InfluxDB]
        Loss --> TS
        Privacy --> TS
        Security --> TS
        Performance --> TS
    end
    
    subgraph "Visualization"
        TS --> Dashboard[Real-time Dashboard]
        Dashboard --> Chart1[📊 Accuracy Chart]
        Dashboard --> Chart2[📉 Loss Chart]
        Dashboard --> Chart3[🔒 Privacy Gauge]
        Dashboard --> Chart4[🛡️ Security Alerts]
        Dashboard --> Chart5[⚡ Performance Metrics]
    end
    
    style Dashboard fill:#e8f5e9
    style Chart1 fill:#e3f2fd
    style Chart2 fill:#e3f2fd
    style Chart3 fill:#fff3e0
    style Chart4 fill:#ffebee
    style Chart5 fill:#f3e5f5
```

---

## Usage Instructions

### Render these diagrams:

1. **GitHub/GitLab**: Automatically renders in `.md` files
2. **VS Code**: Install "Markdown Preview Mermaid Support" extension
3. **Online**: Copy to https://mermaid.live
4. **Documentation**: Use in MkDocs, Docusaurus, or Sphinx

### Customize:

```javascript
// Adjust theme in your markdown file
<script>
  mermaid.initialize({ theme: 'dark' });
</script>
```

---

*Generated for QFLARE Project - November 1, 2025*
