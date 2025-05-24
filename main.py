import cv2
import numpy as np
import os

def run_object_detection(cfg_file, weights_file, obj_file, image_path):
    """
    Performs object detection on an image using a pre-trained YOLOv3 model.

    Args:
        cfg_file (str): Path to the YOLOv3 configuration file (.cfg).
        weights_file (str): Path to the YOLOv3 weights file (.weights).
        obj_file (str): Path to the file containing object class names (.names).
        image_path (str): Path to the input image file.
    """
    # Check if files exist
    if not all(os.path.exists(f) for f in [cfg_file, weights_file, obj_file, image_path]):
        print("Error: One or more required files (cfg, weights, names, or image) do not exist.")
        print(f"Checked: cfg={cfg_file}, weights={weights_file}, obj={obj_file}, image={image_path}")
        return

    # Load object class names
    obj_classes = []
    try:
        with open(obj_file, "r") as f:
            obj_classes = f.read().rstrip("\n").split("\n")
    except Exception as e:
        print(f"Error loading object names file: {e}")
        return

    # Load the YOLOv3 model
    try:
        net = cv2.dnn.readNet(weights_file, cfg_file)
    except cv2.error as e:
        print(f"Error loading YOLO model: {e}")
        print("Please ensure the .cfg and .weights files are valid and compatible with OpenCV's DNN module.")
        return

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}. Check if the path is correct and the image is valid.")
        return

    height, width = image.shape[:2]

    # Pre-process the image for YOLO input
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()
    output_layers = [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers()] # Adjusted for Python indexing

    # Forward pass through the network
    outputs = net.forward(output_layers)

    # Initialize lists for detected objects
    boxes = []
    confidences = []
    class_ids = []

    # Process each output layer
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Adjust confidence threshold as needed
            if confidence > 0.21:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.21, 0.4) # confidence_threshold, nms_threshold

    object_count = 0
    total_rect_area = 0

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(obj_classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            
            # Draw rectangle and label
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {confidence}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            object_count += 1
            total_rect_area += w * h

    # Calculate areas
    image_area = width * height
    remaining_area = image_area - total_rect_area

    # Calculate possible grafts (based on the provided formula)
    # Ensure object_count is not zero to avoid division by zero
    possible_grafts = 0
    if object_count > 0:
        possible_grafts = int((remaining_area * object_count / image_area) + object_count)
    else:
        # If no objects are found, grafts are likely zero.
        # This could be adjusted based on domain knowledge.
        possible_grafts = 0


    # Print results
    print("--- Detection Results ---")
    print(f"Total objects found: {object_count}")
    print(f"Total image area: {image_area} pixels")
    print(f"Total detected object area: {total_rect_area} pixels")
    print(f"Remaining area (image - detected objects): {remaining_area} pixels")
    print(f"تعداد گرافت های احتمالی: {possible_grafts}") # The Persian translation as requested

    # Save the output image
    output_image_path = 'res.png'
    cv2.imwrite(output_image_path, image)
    print(f"Processed image saved to: {output_image_path}")

    # Optionally display the image (will block execution until closed)
    # cv2.imshow('Object Detection', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    # Define paths to your YOLO files and test image
    # IMPORTANT: Ensure these paths are correct relative to where main.py is executed
    # or provide absolute paths.
    
    # Create the '26farvardin.02' directory if it doesn't exist to guide the user
    model_dir = '26farvardin.02'
    test_img_dir = 'Test'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}. Please place your .cfg, .weights, and .names files here.")
        print("Exiting. Please set up model files and run again.")
    elif not os.path.exists(test_img_dir):
        os.makedirs(test_img_dir)
        print(f"Created directory: {test_img_dir}. Please place your test image (e.g., 5.jpg) here.")
        print("Exiting. Please add a test image and run again.")
    else:
        cfg_file = os.path.join(model_dir, 'yolov3_training (5).cfg')
        weights_file = os.path.join(model_dir, 'yolov3_training_last (1).weights')
        obj_file = os.path.join(model_dir, 'obj.names')
        image_path = os.path.join(test_img_dir, '5.jpg')

        run_object_detection(cfg_file, weights_file, obj_file, image_path)
