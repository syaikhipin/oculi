# Third-party licences

Oculi's own code is MIT. The default runtime stack is selected to be commercially usable.

| Component | Version / asset | Licence | Commercial |
|---|---|---|---|
| OpenCV / opencv-python-headless | 5.0 | Apache-2.0 | Yes |
| YuNet face detector | `face_detection_yunet_2023mar.onnx` | MIT (OpenCV Zoo) | Yes |
| SFace face recognition | `face_recognition_sface_2021dec.onnx` | Apache-2.0 (OpenCV Zoo) | Yes |
| YOLOX-s detector | `object_detection_yolox_2022nov.onnx` | Apache-2.0 (OpenCV Zoo) | Yes |
| Silent-Face Anti-Spoofing ONNX | MiniFASNetV2/V1SE | Apache-2.0 | Yes |
| fast-alpr | 0.4 | MIT | Yes |
| open-image-models (plate detector) | MIT repo / weights from releases | MIT | Yes* |
| fast-plate-ocr (CCT OCR) | MIT repo / weights from releases | MIT | Yes* |
| InsightFace buffalo_l | **default** | Code MIT; **weights non-commercial** | **No** |
| ONNX Runtime | 1.31 | MIT | Yes |
| FastAPI / Starlette / Uvicorn | current | MIT / BSD | Yes |
| aiokafka | current | Apache-2.0 | Yes |
| FAISS | current | MIT | Yes |
| SQLite | system | Public domain | Yes |
| Prometheus client | current | Apache-2.0 | Yes |
| Redpanda | container | BSL 1.1 (source-available) | Yes, except competing streaming service |
| Apache Kafka (alternative) | 4.4 | Apache-2.0 | Yes |
| Milestone XProtect | external | Commercial, customer-provided | Customer responsibility |

\* Verify model-weight terms for your jurisdiction before redistribution; the model
repositories are MIT, but always retain their licence notices.

**Do not ship the InsightFace weights commercially without a licence.** Set
`OCULI_FACE_ENGINE=sface` to use the YuNet+SFace path, which avoids that restriction.

Biometric templates are personal data. A commercial deployment needs a lawful basis, retention
policy, access control, and the applicable GDPR/BIPA obligations.
