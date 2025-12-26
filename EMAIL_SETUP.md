# Email Setup Guide

## Configuration Outlook

**Aucune configuration nécessaire !** Le système utilise Outlook installé sur votre ordinateur.

### Installation

Installez simplement la bibliothèque Python :

```bash
pip install pywin32
```

### Prérequis

- Outlook doit être installé et configuré sur votre ordinateur Windows
- Votre compte Outlook doit être configuré dans l'application Outlook
- C'est tout ! Pas besoin de mots de passe ou de configuration dans `.env`

### Tester l'envoi

```bash
python test_email.py
```

Cela enverra un email de test à `nadjari.anthony@gmail.com` uniquement.

**Note :** La première fois, Outlook peut vous demander d'autoriser l'application à envoyer des emails. Cliquez sur "Autoriser".

## Utilisation

### Via le script quotidien

L'email est envoyé automatiquement quand vous exécutez `daily_auto_prediction.py` ou `daily_prediction.bat`.

### Via Streamlit

1. Allez dans l'onglet "Today"
2. Cliquez sur "🔄 Fetch Games"
3. Attendez que les prédictions soient générées
4. Cliquez sur "📧 Envoyer Email"

## Contenu de l'email

L'email contient :
- **Résultats d'hier** : Matchs avec prédictions, cotes, résultats réels et statut (✅/❌)
- **Prédictions d'aujourd'hui** : Matchs prévus avec prédictions, cotes et niveau de confiance

## Destinataires

Les emails sont envoyés à :
- nadjari.anthony@gmail.com
- hugo.dubelloy@hotmail.com

## Dépannage

### "win32com not available"
- Installez pywin32 : `pip install pywin32`
- Redémarrez votre terminal/IDE après installation

### "Outlook is not installed"
- Assurez-vous qu'Outlook est installé sur votre ordinateur Windows
- Le système utilise l'application Outlook installée, pas SMTP

### "Failed to send email"
- Vérifiez que Outlook est ouvert et configuré avec votre compte
- La première fois, Outlook peut demander l'autorisation - cliquez sur "Autoriser"
- Vérifiez que votre compte Outlook est bien configuré dans l'application

### Outlook demande une autorisation
- C'est normal la première fois
- Cliquez sur "Autoriser" ou "Allow" dans la popup Outlook
- Vous pouvez cocher "Ne plus demander" pour les prochaines fois

