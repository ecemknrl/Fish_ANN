Proje Adı: Balık Türlerini Tanıma ve Sınıflandırma (Fish Species Classification)
Proje Açıklaması:
Bu projede, bir veri seti üzerinde balık türlerini sınıflandırmak amacıyla derin öğrenme modelleri kullanılmıştır. Görüntüler öncelikle yeniden boyutlandırılıp normalleştirildikten sonra sıkıştırılmış .npz formatında kaydedilmiştir. Daha sonra bu veriler üzerinde bir Convolutional Neural Network (CNN) modeli eğitilmiştir. Bu README dosyasında, projenin adımları detaylandırılmaktadır.

Adımlar:
1. Veri Ön İşleme:
Veri setindeki balık türleri ve bu türlere ait görüntüler okunmuştur. Bu işlem, klasör yapısına göre organize edilmiştir.
Görüntüler 128x128 piksel boyutuna küçültülmüş ve her görüntü 0-255 aralığındaki pikseller 0-1 aralığına normalize edilmiştir.
Normalleştirilen görüntüler .npz formatında sıkıştırılarak depolanmıştır. Bu sayede depolama alanı tasarrufu sağlanmış ve veri işlemenin hızlandırılması amaçlanmıştır.

***********
for fish_type in fish_types:
    # Balık türüne ait klasördeki alt klasörleri ve dosyaları oku
    for sub_folder in sub_folders:
        # Görüntüleri oku, yeniden boyutlandır ve sıkıştırarak kaydet
        for img_file in os.listdir(image_dir):
            img = Image.open(img_path).convert('RGB').resize(target_size)
            img_array = np.array(img) / 255.0  # Normalizasyon
            np.savez_compressed(output_img_path, img_array)
************

2. Veri Kümesi Ayrıştırma:
Sıkıştırılmış görüntüler ve etiketler bir pandas DataFrame yapısına dönüştürülmüş, bu veri kümesi daha sonra bir CSV dosyasına kaydedilmiştir.
Eğitim ve test veri setleri, %80 eğitim ve %20 test olarak bölünmüştür. Eğitim ve test setleri ayrı CSV dosyaları olarak kaydedilmiştir.
*************
train_data, test_data = train_test_split(data, test_size=0.2, stratify=data['label'])
train_data.to_csv(os.path.join(output_directory, 'train_data.csv'), index=False)
test_data.to_csv(os.path.join(output_directory, 'test_data.csv'), index=False)
**************

3. Model Yapısı:
Projede CNN (Convolutional Neural Network) mimarisi kullanılmıştır. Modelde:
İlk iki katman, sırasıyla 32 ve 64 filtreli Conv2D katmanlarıdır.
Her bir evrişim katmanından sonra bir MaxPooling2D katmanı eklenerek uzamsal boyut küçültülmüş ve özellik çıkarımı yapılmıştır.
Sonrasında bir Flatten katmanı ile veriler düzleştirilmiş ve tam bağlı (Dense) katmanlarla sınıflandırma yapılmıştır.
Aşırı öğrenmeyi önlemek amacıyla Dropout katmanı eklenmiştir.
Çıkış katmanında, balık türlerini sınıflandırmak için softmax aktivasyon fonksiyonu kullanılmıştır.

*****************
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(labels)), activation='softmax')
])

********************

4. Model Eğitimi:
Model Adam optimizasyon algoritması ve sparse_categorical_crossentropy kayıp fonksiyonu ile derlenmiştir.
10 epoch boyunca eğitim gerçekleştirilmiş ve eğitim esnasında doğrulama seti üzerinde modelin performansı değerlendirilmiştir.

***********
history = model.fit(train_images, train_labels, 
                    validation_data=(test_images, test_labels), 
                    epochs=10, batch_size=32)
************
5. Modelin Test Edilmesi ve Sonuçlar:
Model test veri seti üzerinde değerlendirilmiş, kayıp ve doğruluk oranları yazdırılmıştır.
Test seti üzerindeki doğruluk oranı, modelin balık türlerini ne kadar iyi sınıflandırabildiğini gösterir.
*************
6. Sonuçların Görselleştirilmesi:
Eğitim ve doğrulama setleri üzerindeki kayıp ve doğruluk değerleri matplotlib ile görselleştirilmiştir.
Bu grafikler, modelin nasıl ilerlediğini ve öğrenme eğrisini anlamamıza yardımcı olur.

*******************
Dosya Yapısı:
input_directory: Balık veri setinin bulunduğu dizin.
output_directory: Sıkıştırılmış görüntülerin ve etiketlerin kaydedildiği dizin.
train_data.csv: Eğitim veri kümesi.
test_data.csv: Test veri kümesi.
fish_data.csv: Balık türlerinin etiketlendiği ve sıkıştırılmış görüntülerin yollarını içeren veri dosyası.
Kullanılan Teknolojiler:
Python: Temel dil.
TensorFlow/Keras: Derin öğrenme modeli için.
Pandas: Veri işleme.
Pillow (PIL): Görüntü işleme.
Matplotlib: Görselleştirme.
Scikit-learn: Veri ayrıştırma ve model değerlendirme.
*********************
  Sonuç:
Bu projede balık türlerini sınıflandırmak için bir derin öğrenme modeli geliştirilmiş ve başarıyla eğitilmiştir. Model, test veri seti üzerinde doğrulama sonuçlarına göre oldukça başarılı bir performans göstermiştir.
