import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import io

class SignatureVerifier:
    """
    AI-based Signature Verification Engine
    Uses computer vision and machine learning techniques to verify signature authenticity
    """

    def __init__(self):
        self.target_size = (300, 150)

    def preprocess_image(self, image_bytes):
        """
        Preprocess signature image for comparison
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image data")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Remove noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # Find contours to crop signature
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get bounding box of largest contour (the signature)
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(cleaned.shape[1] - x, w + 2 * padding)
            h = min(cleaned.shape[0] - y, h + 2 * padding)

            cropped = cleaned[y:y+h, x:x+w]
        else:
            cropped = cleaned

        # Resize to standard size
        resized = cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_AREA)

        return resized

    def calculate_ssim(self, img1, img2):
        """
        Calculate Structural Similarity Index
        """
        score, diff = ssim(img1, img2, full=True)
        return score

    def calculate_correlation(self, img1, img2):
        """
        Calculate normalized cross-correlation
        """
        # Normalize images
        img1_norm = img1.astype(float) / 255.0
        img2_norm = img2.astype(float) / 255.0

        # Calculate correlation
        correlation = np.corrcoef(img1_norm.flatten(), img2_norm.flatten())[0, 1]
        return correlation

    def calculate_mse(self, img1, img2):
        """
        Calculate Mean Squared Error
        """
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        # Convert to similarity score (0-1)
        similarity = 1 / (1 + mse / 1000)
        return similarity

    def extract_features(self, img):
        """
        Extract features from signature
        """
        # Aspect ratio
        aspect_ratio = img.shape[1] / img.shape[0] if img.shape[0] > 0 else 0

        # Signature density (ratio of black pixels)
        density = np.sum(img > 0) / (img.shape[0] * img.shape[1])

        # Horizontal and vertical projections
        h_projection = np.sum(img, axis=0)
        v_projection = np.sum(img, axis=1)

        # Statistical features
        h_mean = np.mean(h_projection)
        h_std = np.std(h_projection)
        v_mean = np.mean(v_projection)
        v_std = np.std(v_projection)

        return {
            'aspect_ratio': aspect_ratio,
            'density': density,
            'h_mean': h_mean,
            'h_std': h_std,
            'v_mean': v_mean,
            'v_std': v_std
        }

    def compare_features(self, features1, features2):
        """
        Compare extracted features
        """
        # Calculate normalized differences
        aspect_diff = abs(features1['aspect_ratio'] - features2['aspect_ratio']) / max(features1['aspect_ratio'], features2['aspect_ratio'], 0.001)
        density_diff = abs(features1['density'] - features2['density'])

        # Normalize to similarity score
        feature_similarity = 1 - (aspect_diff + density_diff) / 2
        return max(0, min(1, feature_similarity))

    def verify(self, original_bytes, test_bytes):
        """
        Main verification function
        Returns: dict with status, confidence, and details
        """
        try:
            # Preprocess both images
            original_processed = self.preprocess_image(original_bytes)
            test_processed = self.preprocess_image(test_bytes)

            # Calculate multiple similarity metrics
            ssim_score = self.calculate_ssim(original_processed, test_processed)
            correlation_score = self.calculate_correlation(original_processed, test_processed)
            mse_score = self.calculate_mse(original_processed, test_processed)

            # Extract and compare features
            original_features = self.extract_features(original_processed)
            test_features = self.extract_features(test_processed)
            feature_score = self.compare_features(original_features, test_features)

            # Weighted ensemble of metrics
            weights = {
                'ssim': 0.35,
                'correlation': 0.25,
                'mse': 0.20,
                'features': 0.20
            }

            final_score = (
                    ssim_score * weights['ssim'] +
                    correlation_score * weights['correlation'] +
                    mse_score * weights['mse'] +
                    feature_score * weights['features']
            )

            # Determine authenticity (threshold: 0.75)
            threshold = 0.75
            is_genuine = final_score >= threshold

            # Calculate confidence percentage
            confidence = int(final_score * 100)

            return {
                'status': 'GENUINE' if is_genuine else 'FORGED',
                'confidence': confidence,
                'details': {
                    'ssim': round(ssim_score, 4),
                    'correlation': round(correlation_score, 4),
                    'mse': round(mse_score, 4),
                    'features': round(feature_score, 4),
                    'final_score': round(final_score, 4)
                }
            }

        except Exception as e:
            raise Exception(f"Verification failed: {str(e)}")