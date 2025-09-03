import cv2
import numpy as np
import argparse
import os

def detect_shapes(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f'Error: Cannot load image from {image_path}')
        return None, []
    
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    edges = cv2.Canny(blurred, 30, 100)
    combined = cv2.bitwise_or(thresh, edges)
    
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    color_ranges = {
        'red': [
            ((0, 120, 70), (10, 255, 255)), 
            ((170, 120, 70), (180, 255, 255))
        ],
        'green': ((35, 40, 40), (85, 255, 255)),
        'blue': ((90, 50, 50), (130, 255, 255)),
        'yellow': ((20, 100, 100), (40, 255, 255)),
        'orange': ((10, 100, 100), (25, 255, 255)),
        'purple': ((130, 50, 50), (160, 255, 255))
    }
    
    results = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 300 or area > 50000:
            continue
            
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        
        shape = 'unknown'
        
        if vertices == 3:
            shape = 'triangle'
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.85 <= aspect_ratio <= 1.15:
                shape = 'square'
            else:
                shape = 'rectangle'
        elif vertices == 5:
            shape = 'pentagon'
        elif vertices == 6:
            shape = 'hexagon'
        elif vertices > 6:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.75:
                    shape = 'circle'
                elif circularity > 0.6:
                    shape = 'oval'
        
        if shape == 'unknown':
            continue
            
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_val = cv2.mean(hsv, mask=mask)
        h_value, s_value, v_value = mean_val[0], mean_val[1], mean_val[2]
        
        color_name = 'unknown'
        max_score = 0
        
        for color, ranges in color_ranges.items():
            score = 0
            if color == 'red' and isinstance(ranges, list):
                lower1, upper1 = ranges[0]
                lower2, upper2 = ranges[1]
                if ((lower1[0] <= h_value <= upper1[0] and 
                     lower1[1] <= s_value <= upper1[1] and 
                     lower1[2] <= v_value <= upper1[2]) or
                    (lower2[0] <= h_value <= upper2[0] and 
                     lower2[1] <= s_value <= upper2[1] and 
                     lower2[2] <= v_value <= upper2[2])):
                    score = min(s_value, v_value)
            else:
                lower, upper = ranges
                if (lower[0] <= h_value <= upper[0] and 
                    lower[1] <= s_value <= upper[1] and 
                    lower[2] <= v_value <= upper[2]):
                    score = min(s_value, v_value)
            
            if score > max_score:
                max_score = score
                color_name = color
        
        results.append({'shape': shape, 'color': color_name, 'contour': contour, 'area': area, 'vertices': vertices})
        
        cv2.drawContours(original, [contour], -1, (0, 255, 0), 2)
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            text = f'{color_name} {shape}'
            cv2.putText(original, text, (cX-40, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(original, text, (cX-40, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return original, results

def main():
    parser = argparse.ArgumentParser(description='Detect shapes and colors in an image')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('-o', '--output', default='result.jpg', help='Path to save the output image')
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f'Error: File {args.image_path} does not exist')
        print('Available files in current directory:')
        for f in os.listdir('.'):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f'  - {f}')
        return
    
    print(f'Processing image: {args.image_path}')
    result_image, detected_objects = detect_shapes(args.image_path)
    
    if result_image is None:
        return
    
    cv2.imwrite(args.output, result_image)
    print(f'Output image saved as {args.output}')
    
    print('\nDetected Objects:')
    print('-----------------')
    print(f'{"No.":<3} {"Shape":<12} {"Color":<10} {"Vertices":<8} {"Area":<8}')
    print('-' * 40)
    
    for i, obj in enumerate(detected_objects, 1):
        print(f'{i:<3} {obj["shape"]:<12} {obj["color"]:<10} {obj["vertices"]:<8} {obj["area"]:<8.0f}')

if __name__ == '__main__':
    main() 
