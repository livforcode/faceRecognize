from facenet_pytorch import MTCNN, InceptionResnetV1
import sys
import time
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw


class FaceRecognition:
    """人脸识别模块，独立于UI"""

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"运行设备: {self.device}")

        # 初始化模型
        self.mtcnn = MTCNN(
            keep_all=True,
            device=self.device,
            min_face_size=50,
            thresholds=[0.9, 0.9, 0.9],
        )
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # 视频捕获
        self.cap = None

    def open_camera(self):
        """打开摄像头"""
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("无法打开摄像头!")
            return False
        return True

    def process_frame(self):
        """处理视频帧并返回检测结果"""
        if self.cap is None:
            return None, None, None, None

        ret, frame = self.cap.read()
        if not ret:
            return None, None, None, None

        # 转换颜色空间
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # 人脸检测
        boxes, boxes_prob = self.mtcnn.detect(frame_pil)

        if boxes is not None:
            # 提取人脸图像和特征
            face_images = self.mtcnn.extract(
                img=frame_pil, batch_boxes=boxes, save_path=None
            )
            if face_images is not None:
                # 确保输入数据与模型在同一设备上
                face_images = face_images.to(self.device)
                img_embeddings = self.resnet(face_images)
                return boxes, frame_pil, img_embeddings.cpu(), face_images.cpu()

        return None, frame_pil, None, None

    def release(self):
        """释放资源"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class FaceManager:
    """人脸管理器"""

    def __init__(self, conf_thresh=0.75, max_face=10):
        self.max_face = max_face
        self.face_dict = {}  # {id: (特征向量, 出现频率, 标记状态)}
        self.conf_thresh = conf_thresh
        self.face_count = 0

    def find_face(self, face_embedding):
        """查找或添加人脸"""
        if self.face_count == 0:
            self.face_dict[self.face_count] = (face_embedding, 1, 0)
            self.face_count += 1
            return self.face_count - 1, 1, 0

        # 计算相似度
        similarities = {}
        for k, (v, freq, mark) in self.face_dict.items():
            similarities[k] = self.cos_sim(v, face_embedding)

        max_key = max(similarities, key=similarities.get)

        if similarities[max_key] >= self.conf_thresh:
            # 更新现有记录
            old_face, old_freq, old_mark = self.face_dict[max_key]
            self.face_dict[max_key] = (old_face, old_freq + 1, old_mark)
            return max_key, similarities[max_key], old_mark
        else:
            # 添加新人脸
            if len(self.face_dict) < self.max_face:
                new_key = self.face_count
                self.face_dict[new_key] = (face_embedding, 1, 0)
                self.face_count += 1
                return new_key, 1, 0
            else:
                # 替换最少出现的人脸
                lfu_key = min(self.face_dict.items(), key=lambda x: x[1][1])[0]
                del self.face_dict[lfu_key]
                new_key = self.face_count
                self.face_dict[new_key] = (face_embedding, 1, 0)
                self.face_count += 1
                return new_key, 1, 0

    def mark_face(self, face_id, mark):
        """标记人脸状态"""
        if face_id in self.face_dict:
            face, freq, old_mark = self.face_dict[face_id]
            self.face_dict[face_id] = (face, freq, mark)
            return True
        return False

    def get_face_mark(self, face_id):
        """获取人脸标记状态"""
        if face_id in self.face_dict:
            return self.face_dict[face_id][2]
        return 0

    @staticmethod
    def cos_sim(x1, x2):
        """计算余弦相似度"""
        if hasattr(x1, "numpy"):
            x1 = x1.detach().numpy()
        if hasattr(x2, "numpy"):
            x2 = x2.detach().numpy()

        num = float(np.dot(x1, x2))
        denom = np.linalg.norm(x1) * np.linalg.norm(x2)
        return 0.5 + 0.5 * (num / denom) if denom != 0 else 0
