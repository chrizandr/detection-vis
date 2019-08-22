var modal1 = document.getElementById("myModal");
var gimage = document.getElementById("graph");
var modalImg1 = document.getElementById("img01");

gimage.onclick = function(){
  modal1.style.display = "block";
  modalImg1.src = this.src;
}

var close1 = document.getElementById("close1");

close1.onclick = function() {
  modal1.style.display = "none";
}


// Modal 2


var modal2 = document.getElementById("myModal2");
var modalImg2 = document.getElementById("img02");
var close2 = document.getElementById("close2");
var prev = document.getElementById("prev");
var next = document.getElementById("next");

var images = document.getElementsByClassName("result");
var max_idx = images.length;
var curr_idx = 0;
var t_idx = 0;

$(".result").click(function(){
  curr_idx = parseInt(this.id.replace("resimg", ''));
  modal2.style.display = "block";
  modalImg2.src = this.src;
});

close2.onclick = function() {
  modal2.style.display = "none";
}
prev.onclick = function() {
  if (curr_idx != 0){
    curr_idx = curr_idx - 1;
    modalImg2.src = images[curr_idx].src;
  }
}
next.onclick = function() {
  if (curr_idx != max_idx - 1){
    curr_idx = curr_idx + 1;
    modalImg2.src = images[curr_idx].src;
  }
}
