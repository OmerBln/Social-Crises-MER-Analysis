# 🎵 Toplumsal Krizlerin Müzikteki Duygusal Yansımaları  
##  Emotional Reflections of Social Crises in Music


Bu proje, tarihte yaşanmış önemli sosyal ve küresel krizlerin müzik üretimi ve şarkı sözleri üzerindeki duygusal etkilerini incelemek amacıyla geliştirilmiş akademik bir veri bilimi çalışmasıdır.

Savaşlar, ekonomik buhranlar, pandemiler, bölgesel krizler ve toplumsal kırılmalar gibi olayların; belirli dönemlerde üretilen şarkıların liriklerine nasıl yansıdığını **Doğal Dil İşleme (NLP)** ve **Makine Öğrenmesi** teknikleriyle analiz etmeyi hedefler.

---

## 📌 Proje Hakkında

Müzik, yalnızca bir sanat dalı değil; aynı zamanda toplumların ortak psikolojisini, korkularını, umutlarını, kayıplarını ve direnç mekanizmalarını yansıtan güçlü bir kültürel göstergedir.

Bu proje, toplumsal kriz dönemlerinde üretilen şarkı sözlerini veri odaklı biçimde inceleyerek şu soruya cevap arar:

> Tarihsel krizler, toplumun müzikal duygu üretimini ölçülebilir biçimde etkiler mi?

Bu amaç doğrultusunda proje; şarkı sözlerinden duygu çıkarımı yapmakta, yıllara göre duygu dağılımlarını analiz etmekte ve bu dağılımları tarihsel olaylarla ilişkilendirmektedir.

---

## 🎯 Projenin Amacı

Projenin temel amacı, sosyal krizlerin müzik sözleri üzerindeki duygusal yansımalarını matematiksel ve istatistiksel yöntemlerle ortaya koymaktır.

Bu kapsamda hedeflenen başlıca çıktılar şunlardır:

- Tarihsel kriz dönemlerinde üretilen şarkı sözlerinin duygu bakımından analiz edilmesi
- Şarkı sözlerinde yer alan hüzün, öfke, sevgi, korku, endişe ve mutluluk gibi duygu durumlarının sınıflandırılması
- Belirli yıllardaki duygu değişimlerinin tarihsel olaylarla karşılaştırılması
- Kriz dönemlerinde müzik dilinin daha hüzünlü, umutlu, öfkeli veya mutlu hale gelip gelmediğinin incelenmesi
- Veri bilimi, sosyoloji ve müzik araştırmalarını bir araya getiren akademik bir analiz altyapısının oluşturulması

---

## 🧠 Kullanılan Yaklaşım

Proje, yapılandırılmamış müzik verilerinden anlamlı sonuçlar elde etmek için çok aşamalı bir veri işleme hattı kullanır.

Genel işlem akışı şu şekildedir:

1. Şarkı ve sanatçı metadatalarının toplanması  
2. Şarkı sözlerinin veri kaynaklarıyla eşleştirilmesi  
3. Dil filtreleme ve veri temizleme işlemlerinin uygulanması  
4. Liriklerin doğal dil işleme teknikleriyle analiz edilebilir hale getirilmesi  
5. Kelime temsilleri ve makine öğrenmesi modellerinin kullanılması  
6. Duygu sınıflandırması yapılması  
7. Yıllara göre duygu dağılımlarının görselleştirilmesi  
8. Elde edilen sonuçların tarihsel krizlerle karşılaştırılması  

---

## 🔍 Veri Toplama Süreci

Projede güvenilir ve karşılaştırılabilir müzik verileri elde etmek için çok katmanlı bir veri toplama yaklaşımı kullanılmaktadır.

Veri toplama sürecinde:

- **MusicBrainz** üzerinden sanatçı, albüm, yıl ve ülke bilgileri alınır.
- **LRCLIB** üzerinden şarkı sözleriyle eşleştirme yapılır.
- Yerel veri setleri ile eksik veriler desteklenir.
- Sanatçı menşei ve şarkı yılı gibi bilgiler doğrulanır.
- Dil filtreleme işlemiyle analiz dışı kalması gereken şarkılar elenir.

Bu aşama, analiz sonuçlarının güvenilirliği açısından projenin en kritik bölümlerinden biridir.

---

## 🧹 Veri Temizleme ve NLP

Toplanan şarkı sözleri doğrudan modele verilmeden önce çeşitli ön işleme adımlarından geçirilir.

Uygulanan temel işlemler:

- Gereksiz karakterlerin temizlenmesi
- Büyük/küçük harf standardizasyonu
- Noktalama işaretlerinin düzenlenmesi
- Metadata kirliliğinin azaltılması
- Tokenization işlemi
- Kelime sözlüğü oluşturma
- Analize uygun metin formatının hazırlanması

Bu süreç sayesinde ham şarkı sözleri, makine öğrenmesi modelleri tarafından işlenebilir hale getirilir.

---

## 🤖 Modelleme ve Duygu Analizi

Projenin modelleme aşamasında şarkı sözlerinin anlamsal yapısını temsil edebilmek için doğal dil işleme ve makine öğrenmesi teknikleri kullanılmaktadır.

Bu kapsamda:

- Kelimeler arasındaki anlamsal ilişkiler incelenir.
- **Word2Vec** gibi kelime gömme yöntemleriyle kelime vektörleri oluşturulur.
- Şarkı sözlerinden duygu özellikleri çıkarılır.
- Duygu sınıflandırma modelleri kullanılarak şarkıların baskın duygu durumları tahmin edilir.

Amaç, her şarkı için yalnızca yüzeysel kelime analizi yapmak değil; liriklerin taşıdığı genel duygusal atmosferi hesaplamaktır.

---

## 📊 Görselleştirme ve Analiz

Model çıktıları, yıllara göre karşılaştırılabilir grafikler ve istatistiksel çıktılar haline getirilir.

Oluşturulabilecek analiz örnekleri:

- Yıllara göre duygu dağılımı
- Kriz dönemleri öncesi ve sonrası duygu değişimleri
- Üzüntü, umut, öfke veya neşe oranlarının zamansal değişimi
- Belirli olay yıllarında duygu yoğunluğu karşılaştırması
- Sanatçı, ülke veya dönem bazlı duygu eğilimleri

Bu görselleştirmeler, müzik ve toplum ilişkisini daha anlaşılır hale getirmeyi amaçlar.

---

## 🌍 İncelenebilecek Sosyal Kriz Örnekleri

Proje kapsamında farklı tarihsel olaylar analiz edilebilir.

Örnek kriz dönemleri:

- Dünya savaşları
- Ekonomik buhranlar
- 2008 Küresel Ekonomik Krizi
- Bölgesel savaşlar
- Göç krizleri
- Doğal afetler
- 2020 COVID-19 Pandemisi
- Toplumsal protestolar
- Politik ve kültürel kırılma dönemleri

Bu olaylar, ilgili yıllarda üretilen müziklerdeki duygu değişimleriyle karşılaştırılabilir.

---

## 💡 Motivasyon

Sanat, her zaman kendi döneminin en dürüst tanıklarından biridir.

İnsanlık; savaş, pandemi, ekonomik kriz ve toplumsal travma dönemlerinde yaşadığı duyguları yalnızca tarih kitaplarına değil, aynı zamanda şarkılara da kaydetmiştir.

Bu proje, müziğin bu tanıklığını veri bilimi yöntemleriyle inceleyerek insanlığın zor dönemlerdeki psikolojik değişimini ve dayanıklılığını ölçülebilir hale getirmeyi amaçlamaktadır.

---