# Tutorial: License-Plate-Recognition-System

This project implements a **License Plate Recognition System**. It takes an input image of a car,
**processes** it to find the license plate, **isolates** the plate, and then uses
**Optical Character Recognition (OCR)** to read the text on the plate. Finally, it
**displays** the original image with the detected plate and the recognized text.


## Visual Overview

```mermaid
flowchart TD
    A0["Image Loading and Preprocessing
"]
    A1["Contour Detection
"]
    A2["License Plate Localization
"]
    A3["Image Masking and Cropping
"]
    A4["Optical Character Recognition (OCR)
"]
    A5["Result Rendering
"]
    A0 -- "Prepares image for" --> A1
    A1 -- "Provides contours for" --> A2
    A2 -- "Locates area for" --> A3
    A3 -- "Provides image for" --> A4
    A4 -- "Provides text for" --> A5
    A0 -- "Provides original image for" --> A5
    A2 -- "Provides location for" --> A5
```

## Chapters

1. [Image Loading and Preprocessing
](01_image_loading_and_preprocessing_.md)
2. [Contour Detection
](02_contour_detection_.md)
3. [License Plate Localization
](03_license_plate_localization_.md)
4. [Image Masking and Cropping
](04_image_masking_and_cropping_.md)
5. [Optical Character Recognition (OCR)
](05_optical_character_recognition__ocr__.md)
6. [Result Rendering
](06_result_rendering_.md)

---
