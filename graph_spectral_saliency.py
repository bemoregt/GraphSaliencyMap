import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy import sparse
from scipy.sparse.linalg import eigsh
from tkinter import Tk, filedialog

class GraphSpectralSaliency:
    def __init__(self):
        print("Graph Spectral Saliency 초기화 완료")
        
    def compute_saliency(self, image):
        # 이미지를 그레이스케일로 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 이미지를 normalized된 행렬로 변환
        gray = gray.astype(np.float32) / 255.0
        
        # 이미지 크기
        h, w = gray.shape
        
        # 그래프 생성 (격자 그래프)
        # 각 픽셀은 그래프의 노드가 됨
        indices = np.arange(h * w).reshape(h, w)
        
        # 인접 행렬 생성 (희소 행렬로 표현)
        row_indices = []
        col_indices = []
        values = []
        
        # 픽셀 간의 엣지 생성 (4방향 연결: 상, 하, 좌, 우)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for i in range(h):
            for j in range(w):
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        # 픽셀 간의 차이를 가중치로 사용
                        weight = np.exp(-np.abs(gray[i, j] - gray[ni, nj]) / 0.1)
                        
                        row_indices.append(indices[i, j])
                        col_indices.append(indices[ni, nj])
                        values.append(weight)
        
        # 인접 행렬 생성
        adj_matrix = sparse.coo_matrix((values, (row_indices, col_indices)), shape=(h*w, h*w))
        adj_matrix = adj_matrix.tocsr()
        
        # 라플라시안 행렬 계산
        diag = np.array(adj_matrix.sum(axis=1)).flatten()
        diag_matrix = sparse.diags(diag)
        laplacian = diag_matrix - adj_matrix
        
        # 라플라시안 행렬의 고유값과 고유벡터 계산 (가장 작은 고유값 k개)
        k = min(10, h*w-1)  # 사용할 고유벡터 개수
        eigenvalues, eigenvectors = eigsh(laplacian, k=k, which='SM')
        
        # 첫 번째 고유벡터는 상수값이므로 제외하고 두 번째부터 사용
        saliency = np.zeros(h*w)
        
        for i in range(1, k):
            # 고유벡터의 변화량을 기반으로 현저성 계산
            vec = eigenvectors[:, i]
            vec_reshaped = vec.reshape(h, w)
            
            # 그래디언트 계산
            gx, gy = np.gradient(vec_reshaped)
            gradient_magnitude = np.sqrt(gx**2 + gy**2)
            
            # 현저성 값에 추가
            saliency += gradient_magnitude.flatten() * (1.0 / eigenvalues[i])
        
        # 현저성 맵을 이미지 형태로 재구성
        saliency_map = saliency.reshape(h, w)
        
        # 정규화 (0-1 범위로)
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        
        return saliency_map


class SpatialSaliency:
    def __init__(self):
        print("Spatial Saliency 초기화 완료")
    
    def compute_saliency(self, image):
        # 이미지가 흑백인 경우 복사
        if len(image.shape) == 2:
            gray = image.copy()
            # 흑백 이미지를 3채널로 변환 (일부 함수에서 필요)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            # 컬러 이미지를 그레이스케일로 변환
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # 이미지를 Lab 색상 공간으로 변환 (색상 대비 계산을 위해)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # 각 채널 정규화
        l_channel = l_channel.astype(np.float32) / 255.0
        a_channel = a_channel.astype(np.float32) / 255.0
        b_channel = b_channel.astype(np.float32) / 255.0
        
        # 가우시안 필터 적용 (다양한 스케일)
        gfrgb = {}
        for i in range(1, 9, 2):
            sigma = i
            gfrgb[i] = cv2.GaussianBlur(image, (0, 0), sigma)
        
        # 가우시안 필터링된, L, a, b 채널
        gfl, gfa, gfb = {}, {}, {}
        for i in range(1, 9, 2):
            gfl[i] = cv2.GaussianBlur(l_channel, (0, 0), i)
            gfa[i] = cv2.GaussianBlur(a_channel, (0, 0), i)
            gfb[i] = cv2.GaussianBlur(b_channel, (0, 0), i)
        
        # 특징 맵 계산
        # 1. 색상 대비 특징 맵 (L, a, b 채널)
        cfs = np.zeros_like(l_channel)
        
        # 다양한 스케일에서의 색상 대비 계산
        for c in range(1, 9, 2):
            for s in range(1, 9, 2):
                if c != s:
                    # L 채널 대비
                    cfs += np.abs(gfl[c] - gfl[s])
                    # a 채널 대비
                    cfs += np.abs(gfa[c] - gfa[s])
                    # b 채널 대비
                    cfs += np.abs(gfb[c] - gfb[s])
        
        # 2. 밝기 대비 특징 맵
        ifs = np.zeros_like(gray, dtype=np.float32)
        
        # 다양한 스케일에서의 밝기 대비 계산
        for c in range(1, 9, 2):
            for s in range(1, 9, 2):
                if c != s:
                    # 그레이스케일 이미지의 가우시안 블러
                    gc = cv2.GaussianBlur(gray.astype(np.float32) / 255.0, (0, 0), c)
                    gs = cv2.GaussianBlur(gray.astype(np.float32) / 255.0, (0, 0), s)
                    # 밝기 대비 계산
                    ifs += np.abs(gc - gs)
        
        # 3. 방향성 특징 맵 (Gabor 필터 사용)
        ofs = np.zeros_like(gray, dtype=np.float32)
        
        # 다양한 방향으로 Gabor 필터 적용
        for theta in np.arange(0, np.pi, np.pi/4):
            # Gabor 필터 생성
            kernel = cv2.getGaborKernel((21, 21), 5, theta, 10, 0.5, 0, ktype=cv2.CV_32F)
            # 필터 정규화
            kernel /= np.sum(np.abs(kernel))
            # 필터 적용
            filtered = cv2.filter2D(gray.astype(np.float32) / 255.0, -1, kernel)
            # 방향성 특징 맵에 추가
            ofs += filtered
        
        # 모든 특징 맵 결합
        saliency = (cfs + ifs + ofs) / 3.0
        
        # 최종 현저성 맵 후처리
        # 1. 가우시안 필터 적용하여 부드럽게
        saliency = cv2.GaussianBlur(saliency, (0, 0), 3.0)
        
        # 2. 정규화 (0-1 범위로)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency


class SaliencyViewer:
    def __init__(self):
        # 초기화
        self.image = None
        self.spectral_processor = GraphSpectralSaliency()
        self.spatial_processor = SpatialSaliency()
        self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 6))
        
        # 버튼 영역 추가
        self.fig.subplots_adjust(bottom=0.2)
        
        # 이미지 로드 버튼
        ax_load = plt.axes([0.2, 0.05, 0.15, 0.075])
        self.btn_load = Button(ax_load, 'Load Image')
        self.btn_load.on_clicked(self.load_image)
        
        # Graph Spectral Saliency 버튼
        ax_spectral = plt.axes([0.4, 0.05, 0.15, 0.075])
        self.btn_spectral = Button(ax_spectral, 'Graph Spectral Saliency')
        self.btn_spectral.on_clicked(self.compute_spectral_saliency)
        
        # Spatial Saliency 버튼 추가
        ax_spatial = plt.axes([0.6, 0.05, 0.15, 0.075])
        self.btn_spatial = Button(ax_spatial, 'Spatial Saliency Map')
        self.btn_spatial.on_clicked(self.compute_spatial_saliency)
        
        # 초기 텍스트 설정
        self.axs[0].text(0.5, 0.5, 'Load Image', ha='center', va='center', fontsize=12)
        self.axs[1].text(0.5, 0.5, 'Graph Spectral Saliency Map', ha='center', va='center', fontsize=12)
        self.axs[2].text(0.5, 0.5, 'Spatial Saliency Map', ha='center', va='center', fontsize=12)
        
        for ax in self.axs:
            ax.axis('off')
        
        plt.tight_layout()
        print("뷰어 초기화 완료")
    
    def load_image(self, event):
        # tkinter 루트 창을 숨깁니다
        root = Tk()
        root.withdraw()
        
        # 파일 대화상자 직접 호출
        file_path = filedialog.askopenfilename(
            parent=root,
            title="Image Select",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        
        # tkinter 루트 창 파괴
        root.destroy()
        
        if file_path:
            try:
                print(f"이미지 로딩: {file_path}")
                self.image = cv2.imread(file_path)
                
                if self.image is not None:
                    self.display_image()
                else:
                    print("이미지를 로드할 수 없습니다.")
                    self.axs[0].clear()
                    self.axs[0].text(0.5, 0.5, '이미지 로드 실패', ha='center', va='center', fontsize=12)
                    self.axs[0].axis('off')
                    plt.draw()
            except Exception as e:
                print(f"이미지 로드 오류: {e}")
                self.axs[0].clear()
                self.axs[0].text(0.5, 0.5, f'오류: {str(e)}', ha='center', va='center', fontsize=12)
                self.axs[0].axis('off')
                plt.draw()
    
    def compute_spectral_saliency(self, event):
        if self.image is None:
            print("먼저 이미지를 로드해주세요.")
            return
        
        print("Graph Spectral Saliency 맵 계산 중...")
        # 이미지 크기 조정 (계산 속도를 위해)
        max_size = 256
        h, w = self.image.shape[:2]
        scale = min(max_size/h, max_size/w)
        
        if scale < 1:
            new_h, new_w = int(h * scale), int(w * scale)
            resized_image = cv2.resize(self.image, (new_w, new_h))
            print(f"이미지 크기 조정: {h}x{w} -> {new_h}x{new_w}")
        else:
            resized_image = self.image.copy()
        
        # Spectral Saliency 맵 계산
        saliency_map = self.spectral_processor.compute_saliency(resized_image)
        
        # 히트맵 표시
        self.axs[1].clear()
        self.axs[1].imshow(saliency_map, cmap='jet')
        self.axs[1].set_title('Graph Spectral Saliency Map')
        self.axs[1].axis('off')
        plt.draw()
        print("Graph Spectral Saliency 맵 계산 완료")
    
    def compute_spatial_saliency(self, event):
        if self.image is None:
            print("먼저 이미지를 로드해주세요.")
            return
        
        print("Spatial Saliency 맵 계산 중...")
        # 이미지 크기 조정 (계산 속도를 위해)
        max_size = 256
        h, w = self.image.shape[:2]
        scale = min(max_size/h, max_size/w)
        
        if scale < 1:
            new_h, new_w = int(h * scale), int(w * scale)
            resized_image = cv2.resize(self.image, (new_w, new_h))
            print(f"이미지 크기 조정: {h}x{w} -> {new_h}x{new_w}")
        else:
            resized_image = self.image.copy()
        
        # Spatial Saliency 맵 계산
        saliency_map = self.spatial_processor.compute_saliency(resized_image)
        
        # 히트맵 표시
        self.axs[2].clear()
        self.axs[2].imshow(saliency_map, cmap='jet')
        self.axs[2].set_title('Spatial Saliency Map')
        self.axs[2].axis('off')
        plt.draw()
        print("Spatial Saliency 맵 계산 완료")
        
    def display_image(self):
        # 왼쪽 서브플롯 초기화
        self.axs[0].clear()
        
        # BGR -> RGB 변환
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # 이미지 표시
        self.axs[0].imshow(rgb_image)
        self.axs[0].set_title('Original Image')
        self.axs[0].axis('off')
        
        # 가운데 서브플롯 초기화
        self.axs[1].clear()
        self.axs[1].text(0.5, 0.5, 'Press Graph Spectral Saliency button', 
                         ha='center', va='center', fontsize=10)
        self.axs[1].axis('off')
        
        # 오른쪽 서브플롯 초기화
        self.axs[2].clear()
        self.axs[2].text(0.5, 0.5, 'Press Spatial Saliency Map button', 
                         ha='center', va='center', fontsize=10)
        self.axs[2].axis('off')
        
        plt.draw()
        print("원본 이미지 표시 완료")
    
    def show(self):
        plt.show()


if __name__ == "__main__":
    print("애플리케이션 시작")
    viewer = SaliencyViewer()
    viewer.show()