import cv2
import os
from ultralytics import YOLO

# Load your YOLOv8 models
gtsdb_model = YOLO('gtsdb.pt')  # GTSDB model
gtsrb_model = YOLO('gtsrb.pt')  # GTSRB model

def predict_on_frame(input_path, output_path='output_result', mode='gtsdb', conf_threshold=0.6, debug=False):
    """
    Predicts on a given input (image or video).
    
    :param input_path: Path to the input image or video.
    :param output_path: Path to save the output result (without extension).
    :param mode: Which model(s) to use for labeling. Options: 'gtsdb', 'gtsrb', 'both'.
    :param conf_threshold: Minimum confidence threshold to filter predictions.
    :param debug: If True, prints debug logs.
    """
    file_extension = os.path.splitext(input_path)[1].lower()

    if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        # Process as an image
        img = cv2.imread(input_path)
        results = gtsdb_model(img)  # Detect with GTSDB model

        annotated_img = img.copy()  # Create a copy for annotation
        
        for result in results:
            # Extracting bounding boxes and labels from GTSDB model
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_indices = result.boxes.cls.cpu().numpy().astype(int)
            labels = [result.names[idx] for idx in class_indices]

            if debug:
                print("\n--- GTSDB Predictions ---")
            for i, box in enumerate(boxes):
                if scores[i] < conf_threshold:
                    continue  # Skip predictions below the confidence threshold

                x1, y1, x2, y2 = map(int, box)
                if debug:
                    print(f"Box: {box}, Label: {labels[i]}, Confidence: {scores[i]}")

                cropped_sign = img[y1:y2, x1:x2]  # Crop the traffic sign

                if mode == 'gtsrb' or mode == 'both':
                    # Use the GTSRB model to classify the cropped traffic sign
                    gtsrb_results = gtsrb_model(cropped_sign)

                    if debug:
                        print("\n--- GTSRB Predictions for Cropped Sign ---")
                    if isinstance(gtsrb_results, list):
                        for g_result in gtsrb_results:
                            g_boxes = g_result.boxes.xyxy.cpu().numpy()
                            g_scores = g_result.boxes.conf.cpu().numpy()
                            g_class_indices = g_result.boxes.cls.cpu().numpy().astype(int)
                            g_labels = [g_result.names[idx] for idx in g_class_indices]
                            for j, g_box in enumerate(g_boxes):
                                if g_scores[j] < conf_threshold:
                                    continue  # Skip predictions below the confidence threshold
                                if debug:
                                    print(f"Box: {g_box}, Label: {g_labels[j]}, Confidence: {g_scores[j]}")
                    else:
                        g_boxes = gtsrb_results.boxes.xyxy.cpu().numpy()
                        g_scores = gtsrb_results.boxes.conf.cpu().numpy()
                        g_class_indices = gtsrb_results.boxes.cls.cpu().numpy().astype(int)
                        g_labels = [gtsrb_results.names[idx] for idx in g_class_indices]
                        for j, g_box in enumerate(g_boxes):
                            if g_scores[j] < conf_threshold:
                                continue  # Skip predictions below the confidence threshold
                            if debug:
                                print(f"Box: {g_box}, Label: {g_labels[j]}, Confidence: {g_scores[j]}")

                    # If GTSRB does not give a valid label, keep the GTSDB label
                    predicted_label = g_labels[0] if len(g_labels) > 0 else labels[i]
                else:
                    # Default to the label from GTSDB
                    predicted_label = labels[i]

                # Draw bounding box and label
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_img, predicted_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the annotated image
        cv2.imwrite(output_path + '.jpg', annotated_img)
        if debug:
            print(f"\nPrediction saved as {output_path}.jpg")

    elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        # Process as a video
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Prepare the video writer for output
        out = cv2.VideoWriter(output_path + '.mp4', fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = gtsdb_model(frame)  # Detect with GTSDB model
            annotated_frame = frame.copy()  # Create a copy for annotation

            if debug:
                print(f"\n--- Frame {frame_count} ---")
            for result in results:
                # Extracting bounding boxes and labels from GTSDB model
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                class_indices = result.boxes.cls.cpu().numpy().astype(int)
                labels = [result.names[idx] for idx in class_indices]

                if debug:
                    print("--- GTSDB Predictions ---")
                for i, box in enumerate(boxes):
                    if scores[i] < conf_threshold:
                        continue  # Skip predictions below the confidence threshold

                    x1, y1, x2, y2 = map(int, box)
                    if debug:
                        print(f"Box: {box}, Label: {labels[i]}, Confidence: {scores[i]}")

                    cropped_sign = frame[y1:y2, x1:x2]  # Crop the traffic sign

                    if mode == 'gtsrb' or mode == 'both':
                        # Use the GTSRB model to classify the cropped traffic sign
                        gtsrb_results = gtsrb_model(cropped_sign)

                        if debug:
                            print("--- GTSRB Predictions for Cropped Sign ---")
                        if isinstance(gtsrb_results, list):
                            for g_result in gtsrb_results:
                                g_boxes = g_result.boxes.xyxy.cpu().numpy()
                                g_scores = g_result.boxes.conf.cpu().numpy()
                                g_class_indices = g_result.boxes.cls.cpu().numpy().astype(int)
                                g_labels = [g_result.names[idx] for idx in g_class_indices]
                                for j, g_box in enumerate(g_boxes):
                                    if g_scores[j] < conf_threshold:
                                        continue  # Skip predictions below the confidence threshold
                                    if debug:
                                        print(f"Box: {g_box}, Label: {g_labels[j]}, Confidence: {g_scores[j]}")
                        else:
                            g_boxes = gtsrb_results.boxes.xyxy.cpu().numpy()
                            g_scores = gtsrb_results.boxes.conf.cpu().numpy()
                            g_class_indices = gtsrb_results.boxes.cls.cpu().numpy().astype(int)
                            g_labels = [gtsrb_results.names[idx] for idx in g_class_indices]
                            for j, g_box in enumerate(g_boxes):
                                if g_scores[j] < conf_threshold:
                                    continue  # Skip predictions below the confidence threshold
                                if debug:
                                    print(f"Box: {g_box}, Label: {g_labels[j]}, Confidence: {g_scores[j]}")

                        # If GTSRB does not give a valid label, keep the GTSDB label
                        predicted_label = g_labels[0] if len(g_labels) > 0 else labels[i]
                    else:
                        # Default to the label from GTSDB
                        predicted_label = labels[i]

                    # Draw bounding box and label
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, predicted_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Write the annotated frame to the output video
            out.write(annotated_frame)
            frame_count += 1

        # Release the video capture and writer objects
        cap.release()
        out.release()
        if debug:
            print(f"\nPrediction saved as {output_path}.mp4")

    else:
        print("Unsupported file type. Please provide an image or video file.")

# Usage examples:
# For image
predict_on_frame('img1.png', 'output_image', mode='both', debug=True) 
# predict_on_frame('img2.png', 'output_image', mode='both', debug=True) 

# For video
# predict_on_frame('traffic-sign-to-test.mp4', 'ts-ans1', mode='both', debug=False)
