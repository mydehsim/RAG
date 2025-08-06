# 📘 RAG Pipeline - Geliştirilmiş Chainlit Arayüzü
Belge Arama Sistemi için Basitleştirilmiş ve Kararlı Web Arayüzü

**Author: Mustafa Said Oğuztürk**
Tarih: 6 Ağustos 2025
Sürüm: 1.2

## 🚀 Genel Bakış
Bu arayüz, bir RAG (Retrieval-Augmented Generation) sistemini kullanıcı dostu bir web ortamında yönetebilmeniz için tasarlanmıştır.
Amaç, belgeler üzerinde arama yaparken kararlı, anlaşılır ve etkileşimli bir kullanıcı deneyimi sunmaktır.

## 🖥️ Arayüz Özellikleri
**💬 Sohbet Ekranı**
Doğal dilde sorular sorabilir ve belgelerinizden alınan yanıtları görüntüleyebilirsiniz.

Yanıtlarda, alıntılanan kaynakların bilgileri ve tıklanabilir bağlantılar sunulur.

İsteğe bağlı yazma animasyonu ile daha akıcı bir deneyim sağlanır.

**⚙️ Ayarlar Paneli**
Arayüz üzerinden sistem ayarlarını dilediğiniz gibi yapılandırabilirsiniz:

**📂 Yol Ayarları**
Veri Dizini: Belgelerinizin bulunduğu klasörü belirtin.

Veritabanı Dizini: Vektör veritabanının bulunduğu klasör.

**🧠 Model Seçimleri**
Embedding Modeli: Anlamsal temsil (embedding) oluşturmak için kullanılacak model.

LLM Modeli: Yanıtları oluşturacak büyük dil modeli.

İşleme Cihazı: cpu ya da cuda (GPU) seçilebilir.

**🛠️ İşleme Parametreleri**
Chunk Boyutu: Belgelerin parçalara ayrılma büyüklüğü.

Chunk Overlap: Parçalar arasında üst üste binme miktarı.

Arama Sonucu Sayısı: Kaç sonuç getirileceğini belirler.

**🧩 Özellik Aç/Kapa Seçenekleri**
OCR Desteği: Görsellerdeki metinleri okuyabilme özelliği.

Tablo Çıkarma: Belgelerdeki tabloların ayrıştırılması.

İçindekiler Filtresi: İçindekiler sayfası gibi bölümleri filtreleme.

Detaylı Günlük Kaydı: Gelişmiş hata ve işlem günlükleri.

**🖋️ Yazma Animasyonu**
Asistanın yanıtları yavaş yavaş yazmasını sağlar.

**🗂️ Veritabanı Yönetimi**
Veritabanını Güncelle: Yeni eklenen belgeleri dahil eder.

Veritabanını Yeniden Oluştur: Tüm veritabanını sıfırdan oluşturur.

**🔎 Kullanıcı Komutları**
Aşağıdaki komutlar, sohbet ekranında doğrudan yazılarak kullanılabilir:

Komut	Açıklama
/help	Yardım mesajını gösterir
/status	Mevcut veritabanı durumunu gösterir
/docs	Yüklenmiş belgeleri listeler

**📊 Arama ve Yanıtlar**
Bir soru sorduğunuzda sistem:

Belgeleriniz arasında semantik arama yapar.

En uygun içerikleri analiz eder.

Yanıtı üretir ve size sunar.

Kaynak bağlantılarını, chunk ID bilgilerini ve işlem süresini gösterir.

🔗 Kaynaklar, dosya sisteminizdeki konuma doğrudan tıklanabilir bağlantılar olarak sunulur (file:/// biçiminde).

## ℹ️ Ek Bilgiler
Yazma animasyonu tüm Chainlit sürümleriyle uyumludur.

Veritabanı boşsa, arama yapılamaz. Öncelikle veri yolu doğru şekilde tanımlanmalı ve veritabanı güncellenmelidir.

Hatalar kullanıcıya açık ve sade şekilde iletilir.

**📌 Kullanım Amacı**
Bu arayüz; LLM ve belge tabanlı arama sistemleriyle çalışan geliştiricilere, araştırmacılara ve kurumlara:

Belgelerini taramak,

Soru-cevap sistemleri kurmak,

Kaynaklı yanıtlar almak

için sade ve kontrol edilebilir bir çözüm sunar.