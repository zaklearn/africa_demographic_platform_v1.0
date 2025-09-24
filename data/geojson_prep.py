import json
import requests
import os

def download_and_prepare_africa_geojson():
    """
    Télécharger et préparer un GeoJSON personnalisé pour l'Afrique
    qui résout le problème des territoires du Maroc
    """
    
    # Codes ISO3 des 54 pays africains
    african_countries = {
        'DZA': 'Algeria', 'AGO': 'Angola', 'BEN': 'Benin', 'BWA': 'Botswana',
        'BFA': 'Burkina Faso', 'BDI': 'Burundi', 'CMR': 'Cameroon', 'CPV': 'Cabo Verde',
        'CAF': 'Central African Republic', 'TCD': 'Chad', 'COM': 'Comoros', 'COG': 'Congo, Rep.',
        'COD': 'Congo, Dem. Rep.', 'CIV': "Cote d'Ivoire", 'DJI': 'Djibouti',
        'EGY': 'Egypt, Arab Rep.', 'GNQ': 'Equatorial Guinea', 'ERI': 'Eritrea', 'SWZ': 'Eswatini',
        'ETH': 'Ethiopia', 'GAB': 'Gabon', 'GMB': 'Gambia, The', 'GHA': 'Ghana',
        'GIN': 'Guinea', 'GNB': 'Guinea-Bissau', 'KEN': 'Kenya', 'LSO': 'Lesotho',
        'LBR': 'Liberia', 'LBY': 'Libya', 'MDG': 'Madagascar', 'MWI': 'Malawi',
        'MLI': 'Mali', 'MRT': 'Mauritania', 'MUS': 'Mauritius', 'MAR': 'Morocco',
        'MOZ': 'Mozambique', 'NAM': 'Namibia', 'NER': 'Niger', 'NGA': 'Nigeria',
        'RWA': 'Rwanda', 'STP': 'Sao Tome and Principe', 'SEN': 'Senegal',
        'SYC': 'Seychelles', 'SLE': 'Sierra Leone', 'SOM': 'Somalia', 'ZAF': 'South Africa',
        'SSD': 'South Sudan', 'SDN': 'Sudan', 'TZA': 'Tanzania', 'TGO': 'Togo',
        'TUN': 'Tunisia', 'UGA': 'Uganda', 'ZMB': 'Zambia', 'ZWE': 'Zimbabwe'
    }
    
    # Créer le dossier data s'il n'existe pas
    os.makedirs('data', exist_ok=True)
    
    print("📥 Téléchargement du GeoJSON mondial...")
    
    # Télécharger depuis Natural Earth (résolution 50m)
    url = "https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        world_geojson = response.json()
        print("✅ GeoJSON mondial téléchargé")
    except Exception as e:
        print(f"❌ Erreur de téléchargement: {e}")
        return
    
    print("🔧 Filtrage et modification pour l'Afrique...")
    
    # Filtrer les pays africains et modifier le Maroc
    africa_features = []
    morocco_geometries = []
    
    for feature in world_geojson['features']:
        properties = feature['properties']
        
        # Vérifier différents champs possibles pour le code pays
        country_code = None
        for code_field in ['ISO_A3', 'ADM0_A3', 'iso_a3', 'ISO3']:
            if code_field in properties:
                country_code = properties[code_field]
                break
        
        if not country_code:
            # Essayer par nom
            country_name = properties.get('NAME', properties.get('name', ''))
            if country_name == 'Morocco':
                country_code = 'MAR'
        
        if country_code in african_countries:
            # Cas spécial pour le Maroc: fusionner tous les territoires
            if country_code == 'MAR':
                morocco_geometries.append(feature['geometry'])
            else:
                # Normaliser les propriétés
                feature['properties'] = {
                    'ISO_A3': country_code,
                    'NAME': african_countries[country_code],
                    'name': african_countries[country_code]
                }
                africa_features.append(feature)
    
    # Créer une feature unifiée pour le Maroc
    if morocco_geometries:
        print("🇲🇦 Unification des territoires du Maroc...")
        
        # Pour simplifier, on prend la première géométrie du Maroc
        # Dans une version plus sophistiquée, on pourrait fusionner les géométries
        unified_morocco = {
            "type": "Feature",
            "properties": {
                "ISO_A3": "MAR",
                "NAME": "Morocco",
                "name": "Morocco"
            },
            "geometry": morocco_geometries[0]  # Utiliser la géométrie principale
        }
        africa_features.append(unified_morocco)
    
    # Créer le GeoJSON final pour l'Afrique
    africa_geojson = {
        "type": "FeatureCollection",
        "features": africa_features
    }
    
    # Sauvegarder
    output_file = 'data/africa_custom.geojson'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(africa_geojson, f, ensure_ascii=False, indent=2)
    
    print(f"✅ GeoJSON Afrique sauvegardé: {output_file}")
    print(f"📊 {len(africa_features)} pays inclus")
    
    # Vérification
    countries_found = [f['properties']['ISO_A3'] for f in africa_features]
    missing_countries = set(african_countries.keys()) - set(countries_found)
    
    if missing_countries:
        print(f"⚠️  Pays manquants: {missing_countries}")
    else:
        print("✅ Tous les pays africains inclus")

if __name__ == "__main__":
    download_and_prepare_africa_geojson()