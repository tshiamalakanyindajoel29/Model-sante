import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import re

# Charger les stopwords pour le nettoyage du texte
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Fonction de nettoyage du texte
def preprocess_text(text):
    # Mise en minuscule
    text = text.lower()
    # Suppression des URLs et mentions
    text = re.sub(r'http\S+|www\S+|@\S+', '', text)
    # Suppression de la ponctuation et des caractères spéciaux
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization et suppression des stopwords
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Exemple de données textuelles
data = ["The flu season is starting earlier than usual.",
        "High rates of COVID-19 in certain areas.",
        "Vaccination campaigns are effective.",
        "New outbreak of respiratory illness reported."]

# Prétraitement des données
processed_data = [preprocess_text(text) for text in data]

# Convertir le texte en vecteurs numériques
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1000)
data_vectors = vectorizer.fit_transform(processed_data).toarray()

# Mettre à jour la dimension cible à 18 pour correspondre à data_vectors.shape[1]
input_dim = data_vectors.shape[1]
print("Dimension des données :", input_dim)  # Doit être 18

# Paramètres du GAN
latent_dim = 100

# Modèle du générateur
def build_generator():
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(input_dim, activation='sigmoid'))  # correspond à la taille des vecteurs de texte
    model.add(Reshape((input_dim,)))
    return model

# Modèle du discriminateur
def build_discriminator():
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Compilation du GAN
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

# Création du GAN combiné
discriminator.trainable = False
gan = Sequential([generator, discriminator])
gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')
# Entraînement du GAN
def train_gan(data, epochs=1000, batch_size=2):
    for epoch in range(epochs):
        # Entraînement du discriminateur
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]
        fake_data = generator.predict(np.random.normal(0, 1, (batch_size, latent_dim)))
        
        # Entraînement du discriminateur sur données réelles et synthétiques
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
        
        # Moyenne des pertes du discriminateur
        d_loss_real_value = d_loss_real[0] if isinstance(d_loss_real, list) else d_loss_real
        d_loss_fake_value = d_loss_fake[0] if isinstance(d_loss_fake, list) else d_loss_fake
        d_loss = 0.5 * (d_loss_real_value + d_loss_fake_value)

        # Entraînement du générateur
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        g_loss_value = g_loss[0] if isinstance(g_loss, list) else g_loss

        # Affichage des pertes pour chaque 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, D Loss: {d_loss:.4f}, G Loss: {g_loss_value:.4f}")
# Lancer l'entraînement
train_gan(data_vectors, epochs=1000, batch_size=2)

# Générer et visualiser des tendances synthétiques
def generate_trends(n_samples=5):
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    generated_data = generator.predict(noise)
    trends = vectorizer.inverse_transform(generated_data > 0.5)  # Utilisation d'un seuil pour récupérer des mots
    for i, trend in enumerate(trends):
        print(f"Tendance synthétique {i+1}: {' '.join(trend)}")

generate_trends()
