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

$(document).ready(function(){
  for(var i in documents){
    $('#list').append('<label class="checkbox"><input type="checkbox"> ' + documents[i] + '</label><br>');
  }
});
