#!/usr/bin/env python3.11
"""
Script para gerar anotações automáticas usando um modelo YOLO pré-treinado
Este script usa um modelo genérico para detectar objetos e criar anotações iniciais
que podem ser refinadas manualmente depois
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

def auto_anotar_imagens(images_dir, labels_dir, model_path='yolov8n.pt', conf_threshold=0.3):
    """
    Gera anotações automáticas para as imagens
    
    Args:
        images_dir: Diretório com as imagens
        labels_dir: Diretório onde salvar as anotações
        model_path: Caminho do modelo YOLO
        conf_threshold: Threshold de confiança mínimo
    """
    
    print("=" * 60)
    print("AUTO-ANOTAÇÃO DE IMAGENS COM YOLOV8")
    print("=" * 60)
    
    # Carregar modelo
    print(f"\nCarregando modelo: {model_path}")
    model = YOLO(model_path)
    
    # Obter lista de imagens
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    labels_path.mkdir(parents=True, exist_ok=True)
    
    images = list(images_path.glob("*.png")) + list(images_path.glob("*.jpg"))
    
    print(f"Total de imagens para anotar: {len(images)}")
    print(f"Threshold de confiança: {conf_threshold}")
    print(f"\nProcessando...\n")
    
    total_detections = 0
    images_with_detections = 0
    
    for idx, img_path in enumerate(images, 1):
        # Fazer predição
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            verbose=False
        )
        
        # Obter dimensões da imagem
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        # Criar arquivo de anotação
        label_path = labels_path / f"{img_path.stem}.txt"
        
        detections = []
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Obter coordenadas
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Converter para formato YOLO (normalizado)
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                # Classe (0 para todos por enquanto - será refinado manualmente)
                class_id = 0
                
                # Adicionar detecção
                detections.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Salvar anotações
        if detections:
            with open(label_path, 'w') as f:
                f.write('\n'.join(detections))
            images_with_detections += 1
            total_detections += len(detections)
        
        # Mostrar progresso
        if idx % 50 == 0:
            print(f"Processadas: {idx}/{len(images)} imagens")
    
    print("\n" + "=" * 60)
    print("RESUMO DA AUTO-ANOTAÇÃO")
    print("=" * 60)
    print(f"Total de imagens processadas: {len(images)}")
    print(f"Imagens com detecções: {images_with_detections}")
    print(f"Total de detecções: {total_detections}")
    print(f"Média de detecções por imagem: {total_detections/len(images):.2f}")
    print("\n⚠️  IMPORTANTE:")
    print("As anotações geradas são PRELIMINARES e devem ser revisadas manualmente!")
    print("Use ferramentas como Roboflow ou LabelImg para refinar as anotações.")
    print("=" * 60)

def processar_dataset_completo(dataset_dir):
    """
    Processa todos os splits do dataset (train, val, test)
    """
    
    dataset_path = Path(dataset_dir)
    
    for split in ['train', 'val', 'test']:
        images_dir = dataset_path / 'images' / split
        labels_dir = dataset_path / 'labels' / split
        
        if images_dir.exists():
            print(f"\n{'='*60}")
            print(f"Processando split: {split.upper()}")
            print(f"{'='*60}")
            auto_anotar_imagens(images_dir, labels_dir, conf_threshold=0.25)

def main():
    """Função principal"""
    
    print("\n⚠️  AVISO IMPORTANTE ⚠️")
    print("=" * 60)
    print("Este script usa um modelo YOLO genérico pré-treinado no COCO dataset.")
    print("O modelo COCO não foi treinado especificamente para tomates,")
    print("então as detecções podem não ser precisas.")
    print("\nAs anotações geradas são apenas um PONTO DE PARTIDA e devem")
    print("ser REVISADAS E CORRIGIDAS MANUALMENTE antes do treinamento.")
    print("=" * 60)
    
    resposta = input("\nDeseja continuar? (s/n): ")
    
    if resposta.lower() != 's':
        print("Operação cancelada.")
        return
    
    dataset_dir = "/home/ubuntu/dataset_yolo"
    processar_dataset_completo(dataset_dir)
    
    print("\n" + "=" * 60)
    print("PRÓXIMOS PASSOS RECOMENDADOS:")
    print("=" * 60)
    print("1. Revisar as anotações geradas usando Roboflow ou LabelImg")
    print("2. Corrigir bounding boxes incorretas")
    print("3. Adicionar detecções que foram perdidas")
    print("4. Classificar corretamente: ripe (0) ou unripe (1)")
    print("5. Após revisão, iniciar o treinamento do modelo")
    print("=" * 60)

if __name__ == "__main__":
    main()

