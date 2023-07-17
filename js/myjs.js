// window.onload = function(){
// var myTab = document.getElementById("paper-method");    //整个div
// var myUl = myTab.getElementsByTagName("ul")[0];//一个节点
// var myLi = myUl.getElementsByTagName("li");    //数组
// var myDiv = myTab.getElementsByTagName("div"); //数组

// for(var i = 0; i<myLi.length;i++){
//     myLi[i].index = i;
//     myLi[i].onclick = function(){
//         for(var j = 0; j < myLi.length; j++){
//             myLi[j].className = "off";
//             myDiv[j].className = "hide";
//         }
//         this.className = "on";
//         myDiv[this.index].className = "show";
//     }
//   }
// }


let methodTabNavItem = document.querySelectorAll("#method .tab_nav>li");
let methodTabBoxItem = document.querySelectorAll("#method .tab_box>li");
let methodTabNavActive = document.querySelector("#method .tabNav_active");
let methodTabBoxActive = document.querySelector("#method .tabBox_active");

methodTabNavItem.forEach(function(item, index) {
    item.addEventListener('click', function() {
        methodTabNavActive.className = "";
        this.className = "tabNav_active";
        methodTabNavActive = this;

        methodTabBoxActive.className = "";
        methodTabBoxItem[index].className = "tabBox_active";
        methodTabBoxActive = methodTabBoxItem[index];
    }, false)
});


let resultTabNavItem = document.querySelectorAll("#result .tab_nav>li");
let resultTabBoxItem = document.querySelectorAll("#result .tab_box>li");
let resultTabNavActive = document.querySelector("#result .tabNav_active");
let resultTabBoxActive = document.querySelector("#result .tabBox_active");

resultTabNavItem.forEach(function(item, index) {
    item.addEventListener('click', function() {
        resultTabNavActive.className = "";
        this.className = "tabNav_active";
        resultTabNavActive = this;

        resultTabBoxActive.className = "";
        resultTabBoxItem[index].className = "tabBox_active";
        resultTabBoxActive = resultTabBoxItem[index];
    }, false)
});

  
// Automatically setup all `image-comparison` elements after the DOM loads
document.addEventListener('DOMContentLoaded', () => {
    const $els = [...document.querySelectorAll('image-comparison')];
    $els.forEach(function makeComparison($el) {
        const fps = 60;
        const throttleDelay = 1000 / fps;

        $el.onmousemove = updatePosition

        function updatePosition(e) {
            const relative = e.offsetX / $el.clientWidth;
            $el.style.setProperty('--current-position', `${relative * 100}%`);
        }
    });
}, false);



// const slider = document.querySelector(".slider input");
// const img = document.querySelector(".images .img-2");
// const dragLine = document.querySelector(".slider .drag-line");
// slider.oninput = ()=>{
//   let sliderVal = slider.value;
//   dragLine.style.left = sliderVal + "%";
//   img.style.width = sliderVal + "%";
// }


// var cmp1 = document.getElementById("compare-method-ebb-277")
// console.log(cmp1)
// cmp1.style.setProperty('--image1', "url('../img/BLB/bokeh_04_05_ours.jpg')")


var blb_K = 4;
var blb_df = 5;
var blb_left = 'image';
var blb_right = 'ours';
var blb_scene = 277


function getMethodPathBLB(method) {
    var path
    if (method == 'image') {
        path = '../img/BLB/data/' + blb_scene + '/image.jpg'
    }
    else if (method == 'disparity') {
        path = '../img/BLB/data/' + blb_scene + '/disparity.jpg'
    }
    else if (method == 'gt') {
        path = '../img/BLB/data/' + blb_scene + '/bokeh_0' + blb_K + '_0' + blb_df + '.jpg'
    }
    else {
        path = '../img/BLB/' + method + '/' + blb_scene + '/bokeh_0' + blb_K + '_0' + blb_df + '.jpg'
    }
    return path
}



function selectSceneBLB(scene) {
    var select_prev = document.getElementById("select-scene-blb-" + blb_scene)
    var select = document.getElementById("select-scene-blb-" + scene)
    var compare = document.getElementById("compare-method-blb")
    var base = document.getElementById("base-method-blb")
    var path
    
    select_prev.style.opacity = 0.4
    select.style.opacity = 1
    blb_scene = scene

    path = getMethodPathBLB(blb_left)
    compare.style.setProperty('--image1', "url(" + path + ")")
    base.src = path.substring(3)

    path = getMethodPathBLB(blb_right)
    compare.style.setProperty('--image2', "url(" + path + ")")
}


function selectMethodBLB(side) {
    var select = document.getElementById("select-method-blb-" + side)
    var compare = document.getElementById("compare-method-blb")
    var path

    var method = select.options[select.selectedIndex].value
    path = getMethodPathBLB(method)
    if (side == 'left') {
        compare.style.setProperty('--image1', "url(" + path + ")")
        blb_left = method
    }
    else {
        compare.style.setProperty('--image2', "url(" + path + ")")
        blb_right = method
    }
}


function selectK() {
    var select = document.getElementById("select-K")
    var compare = document.getElementById("compare-method-blb")
    var path

    blb_K = select.value
    
    path = getMethodPathBLB(blb_left)
    compare.style.setProperty('--image1', "url(" + path + ")")

    path = getMethodPathBLB(blb_right)
    compare.style.setProperty('--image2', "url(" + path + ")")
}


function selectDf() {
    var select = document.getElementById("select-df")
    var compare = document.getElementById("compare-method-blb")
    var path

    blb_df = select.value
    
    path = getMethodPathBLB(blb_left)
    compare.style.setProperty('--image1', "url(" + path + ")")

    path = getMethodPathBLB(blb_right)
    compare.style.setProperty('--image2', "url(" + path + ")")
}






var ebb_left = 'image';
var ebb_right = 'ours';
var ebb_scene = 218


function getMethodPathEBB(method) {
    var path = '../img/EBB400/' + method + '/' + ebb_scene + '.jpg'
    return path
}



function selectSceneEBB(scene) {
    var select_prev = document.getElementById("select-scene-ebb-" + ebb_scene)
    var select = document.getElementById("select-scene-ebb-" + scene)
    var compare = document.getElementById("compare-method-ebb")
    var base = document.getElementById("base-method-ebb")
    var path
    
    select_prev.style.opacity = 0.4
    select.style.opacity = 1
    ebb_scene = scene

    path = getMethodPathEBB(ebb_left)
    compare.style.setProperty('--image1', "url(" + path + ")")
    base.src = path.substring(3)

    path = getMethodPathEBB(ebb_right)
    compare.style.setProperty('--image2', "url(" + path + ")")
}


function selectMethodEBB(side) {
    var select = document.getElementById("select-method-ebb-" + side)
    var compare = document.getElementById("compare-method-ebb")
    var path

    var method = select.options[select.selectedIndex].value
    path = getMethodPathEBB(method)
    if (side == 'left') {
        compare.style.setProperty('--image1', "url(" + path + ")")
        ebb_left = method
    }
    else {
        compare.style.setProperty('--image2', "url(" + path + ")")
        ebb_right = method
    }
}



var ipb_left = 'image';
var ipb_right = 'ours';
var ipb_scene = '0001'


function getMethodPathIPB(method) {
    var path = '../img/IPB/' + method + '/IMG_' + ipb_scene + '.jpg'
    return path
}



function selectSceneIPB(scene) {
    var select_prev = document.getElementById("select-scene-ipb-" + ipb_scene)
    var select = document.getElementById("select-scene-ipb-" + scene)
    var compare = document.getElementById("compare-method-ipb")
    var base = document.getElementById("base-method-ipb")
    var path
    
    select_prev.style.opacity = 0.4
    select.style.opacity = 1
    ipb_scene = scene

    path = getMethodPathIPB(ipb_left)
    compare.style.setProperty('--image1', "url(" + path + ")")
    base.src = path.substring(3)

    path = getMethodPathIPB(ipb_right)
    compare.style.setProperty('--image2', "url(" + path + ")")
}


function selectMethodIPB(side) {
    var select = document.getElementById("select-method-ipb-" + side)
    var compare = document.getElementById("compare-method-ipb")
    var path

    var method = select.options[select.selectedIndex].value
    path = getMethodPathIPB(method)
    if (side == 'left') {
        compare.style.setProperty('--image1', "url(" + path + ")")
        ipb_left = method
    }
    else {
        compare.style.setProperty('--image2', "url(" + path + ")")
        ipb_right = method
    }
}




// var select = document.getElementById("select-method-ebb-277-left");
// select.onselect = function() { //当选项改变时触发
//     var val = this.options[this.selectedIndex].value; //获取option的value
//      // alert(valOption);
//     // var txtOption = this.options[this.selectedIndex].innerHTML; //获取option中间的文本
//      // alert(txtOption);
//     var path = '../img/BLB/277/' + val + '.jpg'
//     var compare = document.getElementById("compare-method-ebb-277")
//     compare.style.setProperty('--image1', "url(" + path + ")")
// }


// toteg(item) {
//     var id = "#" + item;
//     let idItem = document.getElementById(item);
//     let anchor = this.$el.querySelector(id); //计算传进来的id到顶部的距离
//     this.$nextTick(() => {
//         // console.log(anchor.offsetTop)
//         window.scrollTo(0, anchor.offsetTop - 130);  //滚动距离因为导航栏固定定位130px的高度
//     });
// },

// var anchorLink = document.getElementById("icon-link"),
// target = document.getElementById("dataset");
// anchorLink.addEventListener("click", function(e) {
//     if (window.scrollTo) {
//         e.preventDefault();
//         window.scrollTo({"behavior": "smooth", "top": target.offsetTop});
//     }
// })