(function($) {
  "use strict"; // Start of use strict

  // Smooth scrolling using jQuery easing
  $('a.js-scroll-trigger[href*="#"]:not([href="#"])').click(function() {
    if (location.pathname.replace(/^\//, '') == this.pathname.replace(/^\//, '') && location.hostname == this.hostname) {
      var target = $(this.hash);
      target = target.length ? target : $('[name=' + this.hash.slice(1) + ']');
      if (target.length) {
        $('html, body').animate({
          scrollTop: (target.offset().top - 48)
        }, 1000, "easeInOutExpo");
        return false;
      }
    }
  });

  // Closes responsive menu when a scroll trigger link is clicked
  $('.js-scroll-trigger').click(function() {
    $('.navbar-collapse').collapse('hide');
  });

  // Activate scrollspy to add active class to navbar items on scroll
  $('body').scrollspy({
    target: '#mainNav',
    offset: 54
  });

  // Collapse the navbar when page is scrolled
  $(window).scroll(function() {
    if ($("#mainNav").offset().top > 100) {
      $("#mainNav").addClass("navbar-shrink");
    } else {
      $("#mainNav").removeClass("navbar-shrink");
    }
  });

  // Floating label headings for the contact form
  $(function() {
    $("body").on("input propertychange", ".floating-label-form-group", function(e) {
      $(this).toggleClass("floating-label-form-group-with-value", !!$(e.target).val());
    }).on("focus", ".floating-label-form-group", function() {
      $(this).addClass("floating-label-form-group-with-focus");
    }).on("blur", ".floating-label-form-group", function() {
      $(this).removeClass("floating-label-form-group-with-focus");
    });
  });

  // Show only the first 5 projects by default
  $('.blog-card').each(function(i) {
    if (i < 5) {
      $(this).show();
    } else {
      $(this).hide();
    }
  });

  // Show more project if button is clicked
  $('#show_more_projects').click(function() {
    $('.blog-card').each(function() {
      $(this).show(400);
    });
    // Hide button
    $(this).hide();
  });

  // Hide or show the section if button is clicked
  $('.hr-grey,.hr-light,.hr-primary').click(function() {
    if ($(this).next().is(":visible")) {
      $(this).nextAll().hide();
      $(this).toggleClass('hidden');
    } else {
      $(this).nextAll().show();
      $(this).removeClass('hidden');

      // Show only the first 5 projects
      $(this).nextAll('.blog-card').each(function(i) {
        if (i < 5) {
          $(this).show();
        } else {
          $(this).hide();
        }
      })
      $('#show_more_projects').show();
    }
  });

})(jQuery); // End of use strict
