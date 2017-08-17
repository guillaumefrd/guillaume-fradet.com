var documents = [
  "Carte d'identité",
  "Passeport",
  "Visa et autres papiers d'immigration",
  "Permis de conduire et permis international",
  "Billets aller / retour",
  "Copies de tous les papiers officiels (passeport, permis, billet d’avion)",
  "Lecture",
  "Stylo",
  "Adresses des lieux d'hébergements / Plans"
];

var argent = [
  "Cash / Liquide",
  "Carte de crédit internationale"
];

var sacs = [
  "Valise à roulettes",
  "Sac à dos",
  "Sac étanche",
  "Sac de rando",
  "Cadenas"
];

var vetements = [
  "Parka / veste",
  "K-way",
  "Jean / pantalon",
  "Short",
  "Ceinture",
  "Pull",
  "T-shirts",
  "Chemise",
  "Sous-vêtements",
  "Maillot de bain",
  "Chaussettes",
  "Pyjama",
  "Chaussures de marche",
  "Chaussures de sport",
  "Tongues",
  "Casquette / Bob / Chapeau / Bonnet",
  "Echarpe",
  "Lunettes de soleil",
  "Montre",
  "Serviette"
];

var hygiene = [
  "Trousse de toilette (en soute)",
  "Gel douche (en soute)",
  "Shampooing (en soute)",
  "Dentifrice (en soute)",
  "Brosse à dent",
  "Serviettes",
  "Rasoir",
  "Brosse / peigne",
  "Coupe-ongles / Lime",
  "Déodorant",
  "Crème solaire / après-soleil",
  "Crème hydratante",
  "Pince à épiler",
  "Petit miroir",
  "Lunettes / lentilles",
  "Produit lentilles",
  "Mouchoirs"
];

var sante = [
  "Trousse de secours (désinfectant, pansements, sparadraps, ...)",
  "Paracétamol / ibuprofene",
  "Pillules",
  "Coton",
  "Stick lèvres",
  "Medicaments personnels avec ordonnances"
];

var electro = [
  "Téléphone portable (avec musiques)",
  "Chargeur du téléphone",
  "Appareil photo avec sa housse",
  "Chargeur batterie + batterie secondaire",
  "Carte mémoire",
  "Trepied",
  "GoPro + accessoires",
  "Ordinateur portable",
  "Chargeur d'ordinateur",
  "Disque dur externe",
  "Adaptateur universel pour prises éléctriques",
  "Multiprise"
];

var divers = [
  "Sacs plastiques (linge sale)",
  "Boules Quiés",
  'Masque "anti-lumière"',
  "Jeu de cartes",
  "Gourde",
  "Calculatrice",
  "Parapluie",
  "Clés"
];

$(document).ready(function(){
  for(var i in documents){
    $('#docs').append('<label class="checkbox"><input type="checkbox"> ' + documents[i] + '</label><br>');
  }

  for(var i in argent){
    $('#argent').append('<label class="checkbox"><input type="checkbox"> ' + argent[i] + '</label><br>');
  }

  for(var i in sacs){
    $('#sacs').append('<label class="checkbox"><input type="checkbox"> ' + sacs[i] + '</label><br>');
  }

  for(var i in vetements){
    $('#vetement').append('<label class="checkbox"><input type="checkbox"> ' + vetements[i] + '</label><br>');
  }

  for(var i in hygiene){
    $('#hygiene').append('<label class="checkbox"><input type="checkbox"> ' + hygiene[i] + '</label><br>');
  }

  for(var i in sante){
    $('#sante').append('<label class="checkbox"><input type="checkbox"> ' + sante[i] + '</label><br>');
  }

  for(var i in electro){
    $('#electro').append('<label class="checkbox"><input type="checkbox"> ' + electro[i] + '</label><br>');
  }

  for(var i in divers){
    $('#divers').append('<label class="checkbox"><input type="checkbox"> ' + divers[i] + '</label><br>');
  }
});
