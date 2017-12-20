$(document).ready(function() {

  /* Drawbar */
  $('.barres li').each(function() {
  	var pourcentage = $(this).attr('data-skills');
  	$(this).append($("<span></span>").animate({
  		width : ''+ pourcentage +'%' }, 2000));
  });
});
