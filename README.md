# YOLOv3 Custom Object Detection

This project demonstrates how to perform object detection on an image using a custom-trained YOLOv3 model with OpenCV's DNN module. It loads a pre-trained model, processes an input image, detects objects, calculates areas, and estimates "possible grafts" based on a specific formula.

## Features

-   **Custom YOLOv3 Model Integration**: Uses your own `.cfg`, `.weights`, and `.names` files for object detection.
-   **Image Processing**: Loads an image, performs object detection, and draws bounding boxes around detected objects.
-   **Area Calculation**: Calculates the total area of the image, the combined area of detected objects, and the remaining area.
-   **"Possible Grafts" Estimation**: Applies a custom formula to estimate "possible grafts" based on detection results.
-   **Output Image**: Saves the processed image with detections to `res.png`.

## Prerequisites

Before running the script, ensure you have the following installed:

-   Python 3.6 or higher
-   OpenCV (`opencv-python`)
-   NumPy

## Installation

1.  **Clone this repository (or download the files):**

    ```bash
    git clone [https://github.com/your-username/yolo-object-detection.git](https://github.com/your-username/yolo-object-detection.git)
    cd yolo-object-detection
    ```

2.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Setup and Usage

1.  **Prepare your YOLOv3 Model Files:**
    * Create a directory named `26farvardin.02` in the root of the project (if it doesn't already exist).
    * Place your custom-trained YOLOv3 model files (`.cfg`, `.weights`, and `.names`) inside this `26farvardin.02` directory.
        * `yolov3_training (5).cfg`: Your model configuration file.
        * `yolov3_training_last (1).weights`: Your trained model weights.
        * `obj.names`: A text file listing the names of the classes your model is trained to detect, each on a new line.

    Example `26farvardin.02/` directory:
    ```
    26farvardin.02/
    ├── yolov3_training (5).cfg
    ├── yolov3_training_last (1).weights
    └── obj.names
    ```

2.  **Prepare your Test Image:**
    * Create a directory named `Test` in the root of the project (if it doesn't already exist).
    * Place the image you want to process (e.g., `5.jpg`) inside this `Test` directory.
    * You can rename your image file or update the `image_path` variable in `main.py` accordingly.

    Example `Test/` directory:
    ```
    Test/
    └── 5.jpg
    ```

3.  **Run the script:**

    ```bash
    python main.py
    ```

    The script will load the model, process the image, print detection results to the console, and save the output image with bounding boxes as `res.png` in the project root directory.

## Configuration

You can modify the following paths and parameters in `main.py` (though it's recommended to follow the folder structure):

-   `model_dir`: Directory for YOLO model files (default: `'26farvardin.02'`)
-   `test_img_dir`: Directory for test images (default: `'Test'`)
-   `cfg_file`, `weights_file`, `obj_file`, `image_path`: Full paths to your model and image files.
-   `confidence_threshold`: The minimum confidence to consider a detection valid (currently `0.21`).
-   `nms_threshold`: The Non-Maximum Suppression (NMS) threshold to filter overlapping boxes (currently `0.4`).

## Troubleshooting

-   **"Error: One or more required files do not exist."**: Double-check that your `26farvardin.02` and `Test` directories exist, and that all specified files (`.cfg`, `.weights`, `.names`, and your image) are correctly placed within them and their filenames match exactly as specified in `main.py`.
-   **"Error loading YOLO model..."**: This might indicate an issue with your `.cfg` or `.weights` files. Ensure they are compatible with the OpenCV DNN module.
-   **"Error: Could not load image..."**: Verify the `image_path` is correct and the image file (`5.jpg`) is not corrupted.
-   **No objects detected / Poor detection**:
    * Adjust the `confidence > 0.21` threshold in `main.py`. A lower value might show more detections but also more false positives.
    * Ensure your model (`.weights` file) is well-trained for the objects you are trying to detect.
    * The input image resolution for YOLO is `(416, 416)`. Ensure your objects are discernible at this resolution.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
