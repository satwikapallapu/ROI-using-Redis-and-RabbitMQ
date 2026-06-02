# 🚀 Intelligent Video Analytics Platform using YOLOv7, RabbitMQ & Redis

## 📖 Overview

This project is a distributed real-time video analytics platform designed to process surveillance video streams efficiently using Artificial Intelligence and event-driven architecture.

The system combines custom-trained YOLOv7 object detection, YOLOv7 instance segmentation, RabbitMQ messaging, Redis data storage, and OpenCV-based video processing to create a scalable surveillance solution capable of handling multiple camera feeds simultaneously.

Rather than processing every detected object, the platform incorporates **Region of Interest (ROI) validation** to ensure that only meaningful detections occurring inside predefined monitoring zones are considered valid events. This approach improves accuracy, minimizes false alarms, and reduces storage requirements.

---

# ✨ Key Highlights

*  Real-time multi-camera video processing
*  Custom-trained YOLOv7 object detection model
*  YOLOv7 person segmentation pipeline
*  ROI-based event filtering
*  RabbitMQ producer-consumer architecture
*  Redis-powered configuration and event storage
*  Multi-threaded camera processing
*  Distributed and scalable design
*  Fire & smoke monitoring
*  Vehicle analytics
*  PPE compliance monitoring
*  Human segmentation and occupancy analysis

---

# 🏗️ System Architecture

```text
                  Camera Configuration
                           │
                           ▼
                         Redis
                           │
                           ▼
                    Camera Producer
                           │
                           ▼
                    RabbitMQ Exchange
                           │
          ┌────────────────┴────────────────┐
          │                                 │
          ▼                                 ▼

      Person Queue                   Detection Queue
          │                                 │
          ▼                                 ▼

YOLOv7 Segmentation              YOLOv7 Object Detection
          │                                 │
          └──────────────┬──────────────────┘
                         ▼
                Detection Exchange
                         │
     ┌───────────────────┼───────────────────┐
     │                   │                   │
     ▼                   ▼                   ▼

 Vehicle Queue      Fire Queue       PPE Queue
     │                   │                   │
     ▼                   ▼                   ▼

 ROI Validation    ROI Validation    ROI Validation
     │                   │                   │
     ▼                   ▼                   ▼

                  Redis Event Storage
```

---

# 🧠 Deep Learning Models

## 🎯 Custom Object Detection Model

A custom YOLOv7 model was trained using an annotated dataset containing:

*  Person
*  Car
*  Bike
*  Helmet
*  Jacket
*  Fire
*  Smoke

### 📂 Training Pipeline

1. Dataset Collection
2. Image Annotation using LabelImg
3. YOLOv7 Training & Validation
4. Model Optimization
5. Exporting Trained Weights (`best.pt`)

### 🏷️ Detection Classes

| ID | Class  |
| -- | ------ |
| 0  | Person |
| 1  | Car    |
| 2  | Bike   |
| 3  | Helmet |
| 4  | Jacket |
| 5  | Fire   |
| 6  | Smoke  |

---

# 🔄 Workflow

## 📹 1. Camera Initialization

Camera details are loaded from a centralized configuration source.

Each camera includes:

* Camera ID
* Camera Name
* IP Address
* Video Stream Path
* Detection Labels
* ROI Information

---

## ⚡ 2. Redis Configuration Layer

To reduce repeated file access and improve startup performance, configurations are cached inside Redis.

### Benefits

*  Faster configuration retrieval
*  Centralized management
*  Reduced disk operations
*  Better scalability

---

## 🎞️ 3. Frame Acquisition

Each camera stream is processed independently using multiple threads.

The producer continuously:

* Reads frames
* Resizes images
* Compresses frames
* Encodes data for transmission

---

## 📨 4. RabbitMQ Publishing

Frames are transmitted through RabbitMQ using routing keys.

Depending on camera configuration, frames are routed to:

*  Person Segmentation Queue
*  Object Detection Queue

This enables independent processing pipelines.

---

# 👤 Person Segmentation Pipeline

The segmentation module is dedicated to human segmentation.

### 🔍 Process

1. Receive frame from RabbitMQ
2. Decode image
3. Run YOLOv7 Segmentation
4. Generate person masks
5. Visualize segmented output

### 📤 Output

* Person Masks
* Segmented Human Regions
* Real-Time Visualization

---

# 🤖 Object Detection Pipeline

The detection module performs object recognition using the custom-trained YOLOv7 model.

### 🎯 Supported Objects

* Person
* Car
* Bike
* Helmet
* Jacket
* Fire
* Smoke

### 📊 Detection Output

* Bounding Boxes
* Confidence Scores
* Object Labels
* Timestamped Events

---

# 🔀 Event Routing

Detection results are published to a secondary RabbitMQ exchange and categorized into dedicated queues.

## 🚗 Vehicle Monitoring Queue

Handles:

* Cars
* Bikes

Applications:

* Traffic Analysis
* Parking Monitoring
* Restricted Zone Surveillance

---

## 🔥 Fire & Smoke Queue

Handles:

* Fire
* Smoke

Applications:

* Industrial Safety
* Fire Prevention Systems
* Emergency Alert Generation

---

## 🦺 PPE Monitoring Queue

Handles:

* Helmet
* Jacket

Applications:

* Workplace Compliance
* Construction Site Monitoring
* Industrial Safety Auditing

---

# 📍 ROI-Based Validation

One of the core features of this platform is ROI-based event filtering.

Every detection undergoes spatial validation before being stored.

```text
Object Detected
       │
       ▼
 ROI Validation
       │
 ┌─────┴─────┐
 │           │
 ▼           ▼

Inside      Outside
ROI         ROI
 │           │
 ▼           ▼

Store      Ignore
Event
```

Only events occurring inside configured monitoring zones are considered valid.

### ✅ Benefits

* Reduced False Positives
* Improved Event Relevance
* Lower Storage Consumption
* More Accurate Analytics

---

# 💾 Redis Event Storage

Validated detections are stored inside Redis for rapid retrieval and downstream processing.

### Stored Metadata

* Camera ID
* Timestamp
* Detection Label
* Confidence Score
* Bounding Box Coordinates
* ROI Status
* Encoded Frame Snapshot

---

# 📊 Analytics Modules

### 🚗 Vehicle Analytics

* Vehicle Detection
* Traffic Monitoring
* Zone Occupancy Analysis

### 🔥 Fire Safety Analytics

* Fire Detection
* Smoke Detection
* Incident Monitoring

### 🦺 PPE Compliance Analytics

* Helmet Monitoring
* Jacket Monitoring
* Worker Safety Compliance

### 👥 Human Analytics

* Person Segmentation
* Human Presence Monitoring
* Occupancy Analysis

---

# 🛠️ Technology Stack

| Technology             | Purpose                       |
| ---------------------- | ----------------------------- |
| 🐍 Python              | Core Development              |
| 🤖 YOLOv7              | Object Detection              |
| 👤 YOLOv7 Segmentation | Instance Segmentation         |
| 🎥 OpenCV              | Video Processing              |
| 📨 RabbitMQ            | Distributed Messaging         |
| ⚡ Redis                | Configuration & Event Storage |
| 🏷️ LabelImg           | Dataset Annotation            |
| ⚙️ Multiprocessing     | Parallel Execution            |
| 🧵 Multithreading      | Camera Stream Handling        |

---

# ✨ Project Features

* Distributed AI Architecture
* Real-Time Inference
* Event-Driven Processing
* Multi-Camera Scalability
* ROI-Based Filtering
* Redis Integration
* RabbitMQ Integration
* Modular Design
* Real-Time Visualization
* Custom Model Deployment

---

# 🚀 Future Improvements

*  Multi-Object Tracking (DeepSORT)
*  Intrusion Detection
*  Vehicle Counting
*  Email Notifications
*  SMS Alerts
*  Dashboard Visualization
*  Grafana Monitoring
*  Docker Deployment
*  Kubernetes Deployment
*  AWS / Azure / GCP Deployment

---

# 🌍 Real-World Applications

### 🎥 Smart Surveillance

* Restricted Zone Monitoring
* Automated Threat Detection

### 🏭 Industrial Safety

* PPE Compliance Monitoring
* Worker Safety Analytics

### 🔥 Fire Prevention

* Fire and Smoke Detection
* Early Warning Systems

### 🚗 Traffic Monitoring

* Vehicle Analytics
* Traffic Pattern Analysis

### 🏙️ Smart Cities

* Public Safety Systems
* Intelligent Infrastructure Monitoring

---

# 🏆 Results

This project successfully demonstrates a scalable AI-powered video analytics platform capable of handling multiple camera streams through distributed processing.

By combining custom YOLOv7 models, RabbitMQ messaging, Redis storage, and ROI-based validation, the system delivers efficient event monitoring while minimizing false detections and unnecessary storage overhead.

The architecture is modular, scalable, and suitable for deployment in industrial, surveillance, transportation, and smart city environments.

---

# 📌 Conclusion

The **ROI-Based Intelligent Video Analytics Platform** demonstrates how modern Computer Vision and Distributed Systems can be integrated to build a scalable, high-performance surveillance solution.

By combining **YOLOv7 Detection**, **YOLOv7 Segmentation**, **RabbitMQ**, **Redis**, and **ROI-Based Validation**, the platform efficiently processes multiple video streams while focusing only on meaningful events. This significantly reduces false detections, minimizes storage requirements, and improves overall operational efficiency.

Its modular producer-consumer architecture allows seamless scalability across cameras, analytics modules, and deployment environments, making it suitable for real-world applications such as industrial safety monitoring, smart surveillance, traffic analytics, PPE compliance monitoring, and fire detection systems.

This project showcases the practical implementation of AI-powered video analytics and highlights how distributed computing can be leveraged to build production-ready intelligent monitoring solutions.

---

# 👨‍💻 Author

## Pallapu Sathwika

**Data Science | AI/ML Engineer**

Passionate about building AI-powered solutions in Computer Vision, Machine Learning, Deep Learning, Intelligent Video Analytics, and Distributed Systems.

📧 **Email:** satwikapallapu5@gmail.com

💼 **LinkedIn:** https://www.linkedin.com/in/sathwika-pallapu-a32bba355

⭐ If you found this project useful, consider giving it a Star on GitHub!
