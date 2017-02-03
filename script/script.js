jQuery.ajax = (function(_ajax){

    var protocol = location.protocol,
        hostname = location.hostname,
        exRegex = RegExp(protocol + '//' + hostname),
        YQL = 'http' + (/^https/.test(protocol)?'s':'') + '://query.yahooapis.com/v1/public/yql?callback=?',
        query = 'select * from html where url="{URL}" and xpath="*"';

    function isExternal(url) {
        return !exRegex.test(url) && /:\/\//.test(url);
    }

    return function(o) {

        var url = o.url;

        if ( /get/i.test(o.type) && !/json/i.test(o.dataType) && isExternal(url) ) {

            // Manipulate options so that JSONP-x request is made to YQL

            o.url = YQL;
            o.dataType = 'json';

            o.data = {
                q: query.replace(
                    '{URL}',
                    url + (o.data ?
                        (/\?/.test(url) ? '&' : '?') + jQuery.param(o.data)
                    : '')
                ),
                format: 'xml'
            };

            // Since it's a JSONP request
            // complete === success
            if (!o.success && o.complete) {
                o.success = o.complete;
                delete o.complete;
            }

            o.success = (function(_success){
                return function(data) {

                    if (_success) {
                        // Fake XHR callback.
                        _success.call(this, {
                            responseText: data.results[0]
                                // YQL screws with <script>s
                                // Get rid of them
                                .replace(/<script[^>]+?\/>|<script(.|\s)*?\/script>/gi, '')
                        }, 'success');
                    }

                };
            })(o.success);

        }

        return _ajax.apply(this, arguments);

    };

})(jQuery.ajax);



$.ajax({
    url: 'http://bitcoin.mubiz.com/info',
    type: 'GET',
    success: function(res) {
        var text = res.responseText;
        //var obj = JSON.parse(text);
        //var str = JSON.stringify(text, undefined, 4);
        // then you can manipulate your text as you wish
        //var str = JSON.stringify(text, null, "\t");
        //var obj = JSON.parse(str);
        //str =  JSON.stringify(text, null, "\t");
        //document.getElementsByTagName('')ElementById('info').innerHTML = text;
         document.getElementById('info').innerHTML = text; 
    }
});

$.ajax({
    url: 'http://bitcoin.mubiz.com/blockchaininfo',
    type: 'GET',
    success: function(res) {
        var text = res.responseText;
        // then you can manipulate your text as you wish
        document.getElementById('blockchaininfo').innerHTML = text; 
    }
});

$.ajax({
    url: 'http://bitcoin.mubiz.com/mininginfo',
    type: 'GET',
    success: function(res) {
        var text = res.responseText;
        // then you can manipulate your text as you wish
        document.getElementById('mininginfo').innerHTML = text; 
    }
});

$.ajax({
    url: 'http://bitcoin.mubiz.com/peerinfo',
    type: 'GET',
    success: function(res) {
        var text = res.responseText;
        // then you can manipulate your text as you wish
        document.getElementById('peerinfo').innerHTML = text; 
    }
});

// -------------------
/*
function getJSONP(url, success) {

    var ud = '_' + +new Date,
        script = document.createElement('script'),
        head = document.getElementsByTagName('head')[0] 
               || document.documentElement;

    window[ud] = function(data) {
        head.removeChild(script);
        success && success(data);
    };

    script.src = url.replace('callback=?', 'callback=' + ud);
    head.appendChild(script);

}

getJSONP('http://bitcoin.mubiz.com/info', function(data){
    //console.log(data);
    //document.getElementById('blockchaininfo').innerHTML = data; 
    var str = JSON.stringify(data, undefined, 4);
    output(str);
    //document.body.appendChild(document.createTextNode(JSON.stringify(data, null, 4)));
});  


$.get("http://bitcoin.mubiz.com/info", function(data) {
  console.log(data);
}, 'jsonp');
*/
function output(inp) {
    //document.body.appendChild(document.createElement('pre')).innerHTML = inp;
    document.body.appendChild(document.createTextNode(JSON.stringify(inp, null, 4)));
}
/*
var obj = {a:1, 'b':'foo', c:[false,'false',null, 'null', {d:{e:1.3e5,f:'1.3e5'}}]};
var str = JSON.stringify(obj, undefined, 4);
output(str);
*/

/*
var getJSON = function(url) {
  return new Promise(function(resolve, reject) {
    var xhr = new XMLHttpRequest();
    xhr.open('get', url, true);
    xhr.responseType = 'json';
    xhr.onload = function() {
      var status = xhr.status;
      if (status == 200) {
        resolve(xhr.response);
      } else {
        reject(status);
      }
    };
    xhr.send();
  });
};

getJSON('http://bitcoin.mubiz.com/info').then(function(data) {
    //alert('Your Json result is:  ' + data.result); //you can comment this, i used it to debug

    result.innerText = data.result; //display the result in an HTML element
}, function(status) { //error detection....
  alert('Something went wrong.');
});

*/
/*
$.getJSON('http://bitcoin.mubiz.com/info', function(data){
$('#output').html(data.contents);
});

$.ajax({
        url: 'http://bitcoin.mubiz.com/info',
        type: 'GET',
        dataType: "json",
        success: displayAll
    });

function displayAll(data){
    var str = JSON.stringify(data, undefined, 4);
    document.body.appendChild(document.createElement('pre')).innerHTML = str;
}
*/