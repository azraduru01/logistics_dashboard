# Logistics Dashboard

Streamlit tabanlı lojistik analizi. Uygulama önce MySQL bağlantısını dener; bağlantı kurulamazsa depodaki Delivery_Logistics.csv dosyasını okur.

## Yeniden kurulum

Python sanal ortamı kullanın. Paketlerin tümünün kurulumu bu denetimde çalıştırılmadı; requirements.txt geniş bir ortam dökümüdür ve uyumluluğu ayrıca doğrulanmalıdır.

```sh
git clone https://github.com/azraduru01/logistics_dashboard.git
cd logistics_dashboard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

MySQL kullanacaksanız .env.example dosyasını yalnızca .env henüz yoksa .env adıyla kopyalayın ve kendi bilgilerinizi girin. Mevcut ortam değişkenleri .env değerlerinden önceliklidir. Parola depoya eklenmemelidir. CSV modu canlı MySQL verilerinin yedeği değildir.

## Veritabanı yedeği

MySQL istemcisiyle, parolayı komut satırına yazmadan interaktif girin. Aşağıdaki komut kaynak verileri değiştirmez. Şema adı projedeki varsayılandır; kendi sunucu ve kullanıcı bilgilerinizi kullanın. InnoDB için tutarlı anlık yedek alınırken şema değişikliği yapılmamalıdır.

```sh
mkdir -p backups
umask 077
mysqldump -h localhost -u root -p --single-transaction --routines --triggers --events --databases lojisticks > "backups/lojisticks-$(date +%Y%m%d-%H%M%S).sql"
```

Komutun başarılı bittiğini ve dosyanın boş olmadığını doğrulayın. Yedeği şifreli, proje klasörü dışındaki ikinci bir konuma kopyalayın. Geri yüklemeyi yalnızca ayrı bir test sunucusunda deneyin; dump özgün veritabanı adını içerir. Tablo ve kayıt sayılarını, dashboard sonuçlarını karşılaştırın. Canlı veritabanına geri yükleme yapmayın.

## Denetim notu

Önceki config.py içinde bulunan parola açığa çıkmış kabul edilmeli ve MySQL üzerinde değiştirilmelidir. Bu değişiklik Git geçmişini temizlemez. Yerel dosyalarla tam eşitlik ve yedekten geri dönüş henüz doğrulanmadı; proje klasörünü silmeyin.
