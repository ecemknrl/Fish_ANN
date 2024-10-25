----Fish Species Image Processing and Dataset Compression Project-------
This project was developed as an advanced image processing pipeline tailored for fish species dataset management, incorporating a range of image preprocessing, resizing, normalization, and compression techniques. The primary objective is to enhance the dataset's usability by optimizing image storage, allowing for faster processing in machine learning applications and minimizing storage requirements. This project presents an ideal tool for handling large datasets in machine learning contexts, offering a streamlined approach to dataset preparation, and ultimately aligns with the efficiency standards in data management required in industry settings.

Project Overview
In large-scale machine learning projects, dataset optimization is crucial for both performance and storage management. This project systematically processes a comprehensive fish species dataset to achieve the following:

Image Resizing: Standardizes all images to a target dimension of (128x128), ensuring uniformity across the dataset. This resizing step reduces data dimensionality, leading to faster training and more consistent results.

Normalization: Each pixel value is scaled between 0 and 1 to normalize color distributions, thereby stabilizing the input values and aiding model convergence during training.

Compression: Images are stored in .npz format, which effectively compresses the dataset without compromising data quality. This format enables faster I/O operations and is especially beneficial for large datasets, conserving both storage and memory.

Labeling and Data Logging: All processed files are labeled according to fish species, and a structured CSV log is generated, providing a quick reference for model training and data validation. This log includes the file path and corresponding labels, creating a streamlined method for accessing and analyzing the dataset.

Folder Structure and Organization
The project maintains a clean and organized folder structure. Each fish species has its designated folder, within which each image is processed and stored in a compressed format under the appropriate sub-directory. This approach not only improves dataset readability but also ensures an intuitive layout for future data retrieval and manipulation.

Code Walkthrough
Dataset Loading and Directory Management:

The project reads from a specified input directory containing various fish species in distinct folders.
Subfolders for each species are detected and prepared for organized storage in the output directory, where the compressed images and labels will be saved.
Image Processing Loop:

For each fish species and each image file, the program performs:
Image Opening: Loads each image in RGB format for standardized processing.
Resizing: Resizes the image to 128x128 pixels, ensuring all images maintain a consistent size for model compatibility.

Normalization: Scales pixel values to between 0 and 1, enhancing compatibility with deep learning frameworks.
Compression: Saves each image in .npz format, reducing storage size and optimizing data handling.
CSV File Generation:

Once processing is complete, a CSV file logs each image’s file path and its corresponding label. This structured dataset organization promotes efficient data loading and training in machine learning applications.
Visualization and Exploratory Data Analysis (EDA)
In addition to preprocessing, the project also includes visualizations for understanding dataset characteristics. Key visualizations include:

Distribution of Fish Species: A bar chart showing the frequency of each fish type, aiding in class balance assessment.
Sample Image Grid: A display of randomly selected samples post-processing, providing a visual verification of resizing and normalization.
Pixel Value Distributions: Histograms of normalized pixel intensities, ensuring even distribution across all channels for optimal model input.

Advantages and Practical Applications
This project significantly optimizes dataset handling, making it particularly advantageous for intensive machine learning projects in sectors requiring high computational efficiency, such as financial technology, where large-scale data processing is routine. The organization, preprocessing, and compression aspects of this pipeline demonstrate best practices in data engineering, facilitating faster model training, and reduced storage consumption.

Leveraging these methodologies and automation techniques demonstrates my capability to manage and prepare large datasets, a skill that aligns with data-centric approach to problem-solving in technology-driven environments. This project not only supports seamless integration with deep learning models but also exemplifies a commitment to optimizing resources and ensuring a scalable data processing solution.

Future Enhancements
Automated Augmentation: Introduce data augmentation techniques to further enrich the dataset and improve model robustness.
Dynamic Resizing: Allow for dynamic resizing parameters, enhancing flexibility across various model architectures.
Real-time Processing: Develop a real-time data processing module for deployment within continuous data streams.




*****************************************************************************************************************************************************************************************************

Proje Adı: Balık Türleri Görüntü İşleme ve Veri Seti Sıkıştırma Projesi
Proje Açıklaması:
Bu projede, bir veri seti üzerinde balık türlerini sınıflandırmak amacıyla derin öğrenme modelleri kullanılmıştır. Görüntüler öncelikle yeniden boyutlandırılıp normalleştirildikten sonra sıkıştırılmış .npz formatında kaydedilmiştir. Daha sonra bu veriler üzerinde bir Artificial Neural Network (ANN) modeli eğitilmiştir. Bu README dosyasında, projenin adımları detaylandırılmaktadır.

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
