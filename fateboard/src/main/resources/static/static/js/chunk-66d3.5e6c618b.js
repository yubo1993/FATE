(window.webpackJsonp=window.webpackJsonp||[]).push([["chunk-66d3"],{"9g82":function(t,a,n){"use strict";n.r(a);var e=n("m1cH"),s=n.n(e),o=n("GQeE"),i=n.n(o),r=n("QbLZ"),l=n.n(r),c=n("L2JU"),d=n("Yulh"),h={tooltip:{formatter:"{a} <br/>{b} : {c}%"},grid:{top:"50%"},series:[{title:{show:!1},center:["50%","44%"],name:"value",type:"gauge",detail:{show:!0,color:"#494ece",formatter:"{value}%",offsetCenter:[0,"60%"],fontWeight:"bold",fontSize:36},axisTick:{show:!1,length:4,splitNumber:3,lineStyle:{color:"#fff",opacity:.4}},axisLabel:{distance:-14,color:"#494ece",fontsize:16},axisLine:{lineStyle:{color:[[.25,"rgba(73,78,206,0.7)"],[.5,"rgba(73,78,206,0.8)"],[.75,"rgba(73,78,206,0.9)"],[1,"#494ece"]],width:10}},splitLine:{show:!1},pointer:{show:!0},data:[{value:0,name:"rate"}]}]},p=n("lAiS"),b=n("Xi/h"),u=n("7Qib"),g=n("O8Vv"),f=n("q9c2"),v=n("FyfS"),x=n.n(v);var m=10,w=5,_=10,j="#e8e8ef";function y(t,a,n,e){!(arguments.length>4&&void 0!==arguments[4])||arguments[4]?(t.strokeStyle=function(t,a,n,e){var s=t.createLinearGradient(a.x,a.y,n.x,n.y),o=!0,i=!1,r=void 0;try{for(var l,c=x()(e);!(o=(l=c.next()).done);o=!0){var d=l.value;s.addColorStop(d.pos,d.color)}}catch(t){i=!0,r=t}finally{try{!o&&c.return&&c.return()}finally{if(i)throw r}}return s}(t,{x:a.x,y:a.y},{x:a.x+e,y:a.y},[{pos:0,color:"#494ece"},{pos:1,color:"#24b68b"}]),t.lineWidth=6,t.lineCap="butt"):(t.lineWidth=2,t.lineCap="butt",t.strokeStyle=j);var s=Math.sin(Math.PI/4),o=Math.cos(Math.PI/4),i=a.x+s*n,r=a.y+o*n;return t.beginPath(),t.moveTo(a.x,a.y),t.arc(i,r,n,1.25*Math.PI,1.75*Math.PI),t.stroke(),n}function I(t,a,n,e){return{x:t+e*Math.sin(2*Math.PI*n/360),y:a+e*Math.cos(2*Math.PI*n/360)}}var k={path:function(t,a){var n=a.width,e=a.progress,s=void 0===e?0:e,o=a.content,i=void 0===o?"elapsad":o,r=a.time,l=void 0===r?"00:00:00":r,c=a.x||0,d=(n-2*c)/2/Math.sin(Math.PI/4),h=a.y||d-(n-2*c)/2+6;y(t,{x:c,y:h},d,n),function(t){for(var a=arguments.length>1&&void 0!==arguments[1]?arguments[1]:m,n=arguments.length>2&&void 0!==arguments[2]?arguments[2]:w,e=arguments.length>3&&void 0!==arguments[3]?arguments[3]:_,s=arguments[4],o=arguments[5],i=arguments[6],r=Math.sin(90*Math.PI/360)*s,l=Math.cos(90*Math.PI/360)*s,c=o.x,d=o.y,h=function(o){!function(){var h=o;if(0!==h){var p=Math.sin(2*(45-9*h)*Math.PI/360)*s,b=Math.cos(2*(45-9*h)*Math.PI/360)*s;c+=r-p,d+=l-b,r=p,l=b}var u=I(c,d,45-9*h,a),g=I(u.x,u.y,45-9*h,n),f=I(g.x,g.y,45-9*h,e);t.beginPath(),t.moveTo(u.x,u.y),t.lineTo(g.x,g.y),t.strokeStyle=j,t.lineWidth=1,t.stroke(),t.beginPath(),t.textAlign="center",t.textBaseLine="middle",t.fillStyle="#494ece",t.font=i+"px Arial",t.fillText(10*h,f.x,f.y)}()},p=0;p<11;p++)h(p)}(t,m,w,_,d,{x:c,y:h},10);var p=m+w+_+25;y(t,{x:c+Math.sin(Math.PI/4)*p,y:h+Math.cos(Math.PI/4)*p},d-p,n-Math.sin(Math.PI/4)*p*2,!1);var b=90*s,u=Math.sin(90*Math.PI/360)*d,g=Math.cos(90*Math.PI/360)*d,f=c,v=h,x=I(f+=u-Math.sin(2*(45-b)*Math.PI/360)*d,v+=g-Math.cos(2*(45-b)*Math.PI/360)*d,45-b,p-10),k=I(x.x,x.y,45-b,30);t.beginPath(),t.moveTo(x.x,x.y),t.lineTo(k.x,k.y),t.strokeStyle="#494ece",t.lineWidth=1,t.stroke();var C=c+Math.sin(Math.PI/4)*d,M=h+Math.cos(Math.PI/4)*p-20;t.fillStyle=j,t.font="13px Arial",t.textAlign="center",t.textBaseLine="middle",t.fillText(i,C,M);var S=C,D=h+Math.cos(Math.PI/4)*p;t.fillStyle="#494ece",t.font="15px Arial",t.fillText(l,S,D);var A=C,L=h+Math.cos(Math.PI/4)*p+Math.cos(Math.PI/4)*d/5*2,O=Math.round(100*s)+"%";t.fillStyle="#494ece",t.font="bold 40px Arial",t.fillText(O,A,L)},animations:[{name:"tiktok",callback:function(t){var a=t.time||"00:00:00";(a=a.split(":"))[2]=parseInt(a[2])+1,a[2]>=60&&(a[2]=0,a[1]=parseInt(a[1])+1),a[1]>=60&&(a[1]=0,a[0]=parseInt(a[0])+1),parseInt(a[2])<10&&(a[2]="0"+parseInt(a[2])),parseInt(a[1])<10&&(a[1]="0"+parseInt(a[1])),parseInt(a[0])<10&&(a[0]="0"+parseInt(a[0])),this.setStates({time:a.join(":")})},time:1e3}],events:[{type:"loading",callback:function(t,a,n,e,s){this.addAnimation("loading",function(){if(!(e.progress<s))return!1;e.progress+=.005})}}],clear:function(t,a){t.clearRect(0,0,this.canvas.width,this.canvas.height)}},C={name:"Panel",props:{id:{type:String,default:"panelCanvas"},width:{type:Number,default:700},height:{type:Number,default:1e3},msg:{type:Object,default:function(){return{progress:0,time:0}}}},data:function(){return{widths:this.width,heights:this.height,canvas:null,layer:null,message:this.msg,panelX:0}},watch:{"msg.progress":{handler:function(){this.message=this.msg,this.setProgress(this.message.progress)},deep:!0}},beforeMount:function(){this.initMsg()},mounted:function(){this.calculcateSize(),this.drawing()},methods:{initMsg:function(){this.message.time=this._excahngeTime(this.message.time)},calculcateSize:function(){this.canvas=document.getElementById(this.id);var t=getComputedStyle(this.canvas.parentElement);this.widths=parseInt(t.width.replace("px","")),this.heights=parseInt(t.height.replace("px","")),document.getElementById(this.id).setAttribute("style","width:"+this.widths+"px;height:"+this.heights+"px;"),this.canvas.setAttribute("width",this.widths),this.canvas.setAttribute("height",this.heights),this.panelX=.3*this.widths/2},_excahngeTime:function(t){var a=Math.round(t/1e3),n=a%60,e=(a-n)/60,s=e%24,o=(e-s)/24;return n<10&&(n="0"+n),s<10&&(s="0"+s),o<10&&(o="0"+o),o+":"+s+":"+n},drawing:function(){var t=this.canvas,a=new f.a({canvas:t,state:{width:this.widths,progress:this.message.progress,time:this.message.time,content:"elapsed",x:this.panelX},path:k.path,animations:k.animations,events:k.events,clear:k.clear});this.layer=a},setProgress:function(t){this.layer.emit("loading",{x:0,y:0},t)}}},M=(n("J63Z"),n("KHd+")),S=Object(M.a)(C,function(){var t=this.$createElement,a=this._self._c||t;return a("div",[a("canvas",{staticStyle:{display:"block"},attrs:{id:this.id}})])},[],!1,null,"2d9bb583",null);S.options.__file="index.vue";var D=S.exports,A=n("dv4G"),L={components:{EchartContainer:d.a,Dag:g.a,Panel:D},data:function(){return{jobOptions:h,graphOptions:p.a,datasetList:[],jobId:this.$route.query.job_id,role:this.$route.query.role,partyId:this.$route.query.party_id,jobStatus:"",jobDetail:{},datasetLoading:!0,logLoading:!1,jobTimer:null,logWebsocket:{error:null,warning:null,info:null,debug:null},jobWebsocket:null,logsMap:{error:{list:[],length:0},warning:{list:[],length:0},info:{list:[],length:0},debug:{list:[],length:0}},DAGData:null,gaugeInstance:null,graphInstance:null,ratio:"",count:"",AUC:"",elapsed:"",currentLogTab:"info",showGraph:!1}},computed:l()({},Object(c.b)(["icons"])),mounted:function(){this.getDatasetInfo(),this.getDAGDpendencies(),this.getLogSize(),this.openLogsWebsocket(),this.openJobWebsocket()},beforeDestroy:function(){clearInterval(this.jobTimer),this.closeWebsocket()},methods:{getDAGDpendencies:function(){var t=this,a={job_id:this.jobId,role:this.role,party_id:this.partyId};Object(A.d)(a).then(function(a){t.DAGData=a.data})},openLogsWebsocket:function(){var t=this;i()(this.logsMap).forEach(function(a){t.logWebsocket[a]=Object(u.e)("/log/"+t.jobId+"/"+t.role+"/"+t.partyId+"/default/"+a,function(t){},function(n){var e=JSON.parse(n.data);Array.isArray(e)?e.length>0&&(t.logsMap[a].list=[].concat(s()(t.logsMap[a].list),s()(e)),t.logsMap[a].length=e[e.length-1].lineNum):(t.logsMap[a].list.push(e),t.logsMap[a].length=e.lineNum)})})},openJobWebsocket:function(){var t=this,a=this;this.jobWebsocket=Object(u.e)("/websocket/progress/"+this.jobId+"/"+this.role+"/"+this.partyId,function(t){},function(n){var e=JSON.parse(n.data),s=e.process,o=e.status,i=e.duration,r=e.dependency_data;t.graphInstance&&t.pushDataToGraphInstance(t.graphInstance,r.data),i&&(t.elapsed=Object(u.d)(i)),t.jobStatus=o,"failed"!==t.jobStatus&&"success"!==t.jobStatus&&(t.jobOptions.series[0].pointer.show=!0,t.jobOptions.series[0].detail.show=!0,t.jobOptions.series[0].data[0].value=s||0),t.gaugeInstance&&t.gaugeInstance.setOption(t.jobOptions,!0),a.jobDetail.progress=s||0})},getLogSize:function(){},getDatasetInfo:function(){var t=this,a={job_id:this.jobId,role:this.role,party_id:this.partyId};Object(A.e)(a).then(function(a){var n=a.data,e=n.job,s=n.dataset;if(s){var o=s.roles,r=s.dataset,l=[];i()(o).forEach(function(t){var a=[];o[t].forEach(function(t){a.push({value:t,label:t})});var n={role:t.toUpperCase(),options:a,roleValue:a[0].label,datasetData:r[t]||""};l.push(n)}),t.datasetList=l.sort(function(t,a){var n=t.role,e=a.role;return"GUEST"===e?1:"HOST"===e&&"ARBITER"===n?1:void 0})}e&&(t.jobStatus=e.fStatus,t.jobDetail={progress:e.fProgress,time:e.fElapsed})}).then(function(a){t.datasetLoading=!1})},getJobEchartInstance:function(t){this.gaugeInstance=t},closeWebsocket:function(){var t=this;i()(this.logWebsocket).forEach(function(a){t.logWebsocket[a]&&t.logWebsocket[a].close()}),this.jobWebsocket&&this.jobWebsocket.close()},handleKillJob:function(t){var a=this,n={job_id:this.jobId,role:this.role,party_id:this.partyId};Object(A.e)(n).then(function(e){var s=e.data.job;"waiting"===(a.jobStatus=s.fStatus)?t="cancel":"cancel"===t&&(t="kill"),a.killJob(n,t)})},killJob:function(t,a){var n=this;this.$confirm("You can't undo this action', 'Are you sure you want to "+a+" this job?",{confirmButtonText:"Sure",cancelButtonText:"Cancel"}).then(function(){Object(A.f)(t).then(function(){n.getDatasetInfo(),n.getDAGDpendencies()})}).catch(function(){})},getGraphEchartInstance:function(t){var a=this,n=null;n=window.setInterval(function(){a.DAGData&&(window.clearInterval(n),a.graphOptions.tooltip.show=!1,a.pushDataToGraphInstance(t,a.DAGData))},100),this.graphInstance=t},pushDataToGraphInstance:function(t,a){var n=Object(b.a)(a),e=n.dataList,s=n.linksList;this.graphOptions.series[0].data=e,this.graphOptions.series[0].links=s,t.setOption(this.graphOptions,!0)},toDetails:function(){this.$router.push({path:"/details",query:{job_id:this.jobId,role:this.role,party_id:this.partyId,from:"Dashboard"}})},switchLogTab:function(t){this.currentLogTab=t},logOnMousewheel:function(t){var a=this,n=this.logsMap[this.currentLogTab].list[0];if(n){var e=n.lineNum-1;if(e>0){if(0===this.$refs.logView.scrollTop&&(t.wheelDelta>0||t.detail>0)){var o=e-1e3>1?e-1e3:1;if(!this.logLoading){this.logLoading=!0;window.setTimeout(function(){Object(A.h)({componentId:"default",job_id:a.jobId,role:a.role,party_id:a.partyId,begin:o,end:e,type:a.currentLogTab}).then(function(t){var n=[];t.data.map(function(t){t&&n.push(t)}),a.logsMap[a.currentLogTab].list=[].concat(n,s()(a.logsMap[a.currentLogTab].list)),a.logLoading=!1}).catch(function(){a.logLoading=!1})},1e3)}}}}}}},O=(n("lJfc"),Object(M.a)(L,function(){var t=this,a=t.$createElement,n=t._self._c||a;return n("div",{staticClass:"dashboard-container bg-dark app-container"},[n("h3",{staticClass:"app-title flex space-between"},[n("span",[t._v("Dashboard")]),t._v(" "),n("p",[t._v("Job: "),n("span",[t._v(t._s(t.jobId))])])]),t._v(" "),n("el-row",{staticClass:"dash-board-list",attrs:{gutter:24}},[n("el-col",{attrs:{span:8}},[n("div",{directives:[{name:"loading",rawName:"v-loading",value:t.datasetLoading,expression:"datasetLoading"}],staticClass:"col dataset-info shadow"},[n("h3",{staticClass:"list-title",staticStyle:{"margin-bottom":"24px"}},[t._v("DATASET INFO")]),t._v(" "),t._l(t.datasetList,function(a,e){return n("el-row",{key:e,staticClass:"dataset-row",attrs:{gutter:4}},[n("el-col",{attrs:{span:6,offset:2}},[n("div",{staticClass:"dataset-item"},[n("p",{staticClass:"name dataset-title"},[t._v(t._s(a.role))]),t._v(" "),1===a.options.length?n("p",{staticClass:"value"},[t._v(t._s(a.roleValue))]):n("el-select",{model:{value:a.roleValue,callback:function(n){t.$set(a,"roleValue",n)},expression:"row.roleValue"}},[t._l(a.options,function(t,a){return n("el-option",{key:a,attrs:{value:t.value,label:t.label}})}),t._v("\n                "+t._s(a.roleValue)+"\n              ")],2)],1)]),t._v(" "),n("el-col",{attrs:{span:14}},[n("div",{staticClass:"dataset-item"},[n("p",{staticClass:"name"},[t._v("DATASET")]),t._v(" "),n("p",{staticClass:"value"},[n("el-tooltip",{attrs:{content:a.datasetData?Object.values(a.datasetData[a.roleValue]).join(", "):"",placement:"top"}},[n("span",[t._v(t._s(a.datasetData?Object.values(a.datasetData[a.roleValue]).join(", "):""))])])],1)])])],1)})],2)]),t._v(" "),n("el-col",{attrs:{span:8}},[n("div",{staticClass:"col job flex-center justify-center shadow pos-r"},[n("h3",{staticClass:"list-title"},[t._v("JOB")]),t._v(" "),"failed"===t.jobStatus||"success"===t.jobStatus?n("div",{staticClass:"job-end-container flex flex-col flex-center"},["failed"===t.jobStatus?n("img",{staticClass:"job-icon",attrs:{src:t.icons.normal.failed,alt:""}}):t._e(),t._v(" "),"success"===t.jobStatus?n("img",{staticClass:"job-icon",attrs:{src:t.icons.normal.success,alt:""}}):t._e(),t._v(" "),n("ul",{staticClass:"job-info flex space-around flex-wrap w-100"},[n("li",[n("p",{staticClass:"name"},[t._v("status")]),t._v(" "),n("p",{staticClass:"value"},[t._v(t._s(t.jobStatus))])]),t._v(" "),t.elapsed?n("li",[n("p",{staticClass:"name"},[t._v("duration")]),t._v(" "),n("p",{staticClass:"value"},[t._v(t._s(t.elapsed))])]):t._e(),t._v(" "),t.AUC?n("li",[n("p",{staticClass:"name overflow-ellipsis"},[t._v("best score(AUC)")]),t._v(" "),n("p",{staticClass:"value"},[t._v(t._s(t.AUC))])]):t._e(),t._v(" "),t.ratio?n("li",[n("p",{staticClass:"name"},[t._v("ratio")]),t._v(" "),n("p",{staticClass:"value"},[t._v(t._s(t.ratio))])]):t._e(),t._v(" "),t.count?n("li",[n("p",{staticClass:"name"},[t._v("count")]),t._v(" "),n("p",{staticClass:"value"},[t._v(t._s(t.count))])]):t._e()])]):"waiting"===t.jobStatus||"running"===t.jobStatus?n("div",{staticClass:"echarts-container"},[t.elapsed?n("div",{staticClass:"elapsed"},[n("p",{staticClass:"elapsed-title"},[t._v("elapsed")]),t._v(" "),n("p",{staticClass:"elapsed-time text-primary"},[t._v(t._s(t.elapsed))])]):t._e(),t._v(" "),n("echart-container",{class:"echarts",attrs:{options:t.jobOptions},on:{getEchartInstance:t.getJobEchartInstance}})],1):t._e(),t._v(" "),n("div",{staticClass:"btn-wrapper flex flex-col flex-center pos-a"},[n("p",{directives:[{name:"show",rawName:"v-show",value:"running"===t.jobStatus||"waiting"===t.jobStatus,expression:"jobStatus==='running' || jobStatus==='waiting'"}],staticClass:"kill text-primary pointer",on:{click:function(a){t.handleKillJob("running"===t.jobStatus?"kill":"cancel")}}},[t._v(t._s("running"===t.jobStatus?"kill":"cancel"))]),t._v(" "),n("el-button",{attrs:{type:"primary",round:""},on:{click:function(a){t.toDetails(t.jobId)}}},[t._v("view this job\n          ")])],1)])]),t._v(" "),n("el-col",{attrs:{span:8}},[n("div",{directives:[{name:"loading",rawName:"v-loading",value:!1,expression:"false"}],staticClass:"col graph flex-center justify-center shadow"},[n("h3",{staticClass:"list-title"},[t._v("GRAPH")]),t._v(" "),t.DAGData?n("div",{staticClass:"wrapper w-100 pointer",staticStyle:{"min-width":"400px"},style:{"min-height":60*t.DAGData.component_list.length+"px"},on:{click:function(a){t.showGraph=!0}}},[n("dag",{attrs:{id:"graphCanvas",msg:t.DAGData}})],1):t._e()])])],1),t._v(" "),n("div",{staticClass:"log-wrapper shadow"},[n("div",{staticClass:"flex flex-center",staticStyle:{padding:"18px 0"}},[n("h3",{staticClass:"title"},[t._v("LOG")]),t._v(" "),n("ul",{staticClass:"tab-bar flex"},t._l(Object.keys(t.logsMap),function(a,e){return n("li",{key:e,staticClass:"tab-btn",class:{"tab-btn-active":t.currentLogTab===a},on:{click:function(n){t.switchLogTab(a)}}},[n("span",{staticClass:"text"},[t._v(t._s(a))]),t._v(" "),"all"!==a?n("span",{staticClass:"count",class:[a]},[t._v(t._s(t.logsMap[a].length))]):t._e()])}))]),t._v(" "),n("div",{directives:[{name:"loading",rawName:"v-loading",value:t.logLoading,expression:"logLoading"}],ref:"logView",staticClass:"log-container",on:{mousewheel:t.logOnMousewheel}},[n("ul",{staticClass:"log-list overflow-hidden"},t._l(t.logsMap[t.currentLogTab].list,function(a,e){return n("li",{key:e},[n("div",{staticClass:"flex"},[n("span",{staticClass:"line-num"},[t._v(t._s(a.lineNum))]),t._v(" "),n("span",{staticClass:"content"},[t._v(" "+t._s(a.content))])])])}))])]),t._v(" "),n("el-dialog",{attrs:{visible:t.showGraph,"close-on-click-modal":!1,width:"80%",top:"10vh"},on:{"update:visible":function(a){t.showGraph=a}}},[t.DAGData?n("div",{staticClass:"wrapper w-100",staticStyle:{"min-width":"400px"},style:{"min-height":60*t.DAGData.component_list.length+"px"}},[n("dag",{attrs:{id:"dialogCanvas",msg:t.DAGData}})],1):t._e()])],1)},[],!1,null,null,null));O.options.__file="index.vue";a.default=O.exports},FqAf:function(t,a,n){var e=n("zH3H");"string"==typeof e&&(e=[[t.i,e,""]]),e.locals&&(t.exports=e.locals);(0,n("SZ7m").default)("9b71280a",e,!0,{})},J63Z:function(t,a,n){"use strict";var e=n("FqAf");n.n(e).a},lJfc:function(t,a,n){"use strict";var e=n("p9cb");n.n(e).a},p9cb:function(t,a,n){var e=n("xWUn");"string"==typeof e&&(e=[[t.i,e,""]]),e.locals&&(t.exports=e.locals);(0,n("SZ7m").default)("01df6e23",e,!0,{})},xWUn:function(t,a,n){(t.exports=n("I1BE")(!1)).push([t.i,".dashboard-container {\n  height: 100%;\n}\n.dashboard-container .dash-board-list .col {\n    height: 360px;\n    background: #fff;\n    border-radius: 4px;\n}\n.dashboard-container .dash-board-list .col .list-title {\n      height: 40px;\n      padding-top: 20px;\n      font-size: 18px;\n      color: #534c77;\n      text-indent: 24px;\n}\n.dashboard-container .dash-board-list .col .echarts-container {\n      width: 100%;\n      height: 100%;\n      position: relative;\n}\n.dashboard-container .dash-board-list .dataset-info {\n    -webkit-box-orient: vertical;\n    -webkit-box-direction: normal;\n        -ms-flex-direction: column;\n            flex-direction: column;\n}\n.dashboard-container .dash-board-list .dataset-info .dataset-row {\n      margin-top: 30px;\n}\n.dashboard-container .dash-board-list .dataset-info .dataset-row:first-of-type {\n        margin-top: 32px;\n}\n.dashboard-container .dash-board-list .dataset-info .dataset-row .dataset-item {\n        margin-bottom: 20px;\n}\n.dashboard-container .dash-board-list .dataset-info .dataset-row .dataset-item .name {\n          margin-bottom: 6px;\n          color: #bbbbc8;\n}\n.dashboard-container .dash-board-list .dataset-info .dataset-row .dataset-item .dataset-title {\n          font-weight: bold;\n          color: #7f7d8e;\n}\n.dashboard-container .dash-board-list .dataset-info .dataset-row .dataset-item .value {\n          font-size: 16px;\n          color: #7f7d8e;\n          font-weight: bold;\n          overflow: hidden;\n          white-space: nowrap;\n          text-overflow: ellipsis;\n}\n.dashboard-container .dash-board-list .job .echarts {\n    width: 100%;\n    height: 320px;\n    /*top: 5%;*/\n}\n.dashboard-container .dash-board-list .job .elapsed {\n    position: absolute;\n    top: 0;\n    right: 16px;\n}\n.dashboard-container .dash-board-list .job .elapsed .elapsed-title {\n      margin-right: 14px;\n      margin-bottom: 6px;\n      color: #bbbbc8;\n      text-align: right;\n}\n.dashboard-container .dash-board-list .job .elapsed .elapsed-time {\n      height: 28px;\n      width: 88px;\n      background: #f8f8fa;\n      text-align: center;\n      line-height: 28px;\n      border-radius: 28px;\n      font-size: 16px;\n}\n.dashboard-container .dash-board-list .job .job-end-container {\n    height: 320px;\n}\n.dashboard-container .dash-board-list .job .job-end-container .job-icon {\n      margin-top: 35px;\n      margin-bottom: 36px;\n      font-size: 50px;\n}\n.dashboard-container .dash-board-list .job .job-end-container .job-info {\n      padding: 0 10px;\n}\n.dashboard-container .dash-board-list .job .job-end-container .job-info > li {\n        width: 28%;\n        margin-bottom: 20px;\n}\n.dashboard-container .dash-board-list .job .job-end-container .job-info > li .name {\n          font-size: 14px;\n          color: #bbbbc8;\n}\n.dashboard-container .dash-board-list .job .job-end-container .job-info > li .value {\n          font-size: 16px;\n          font-weight: bold;\n          color: #7f7d8e;\n}\n.dashboard-container .dash-board-list .job .btn-wrapper {\n    position: absolute;\n    left: 0;\n    bottom: 32px;\n    width: 100%;\n}\n.dashboard-container .dash-board-list .job .btn-wrapper .kill:hover {\n      text-decoration: underline;\n}\n.dashboard-container .dash-board-list .job .btn-wrapper .el-button {\n      padding: 7px 24px 6px;\n      margin-top: 6px;\n}\n.dashboard-container .graph {\n    overflow: auto;\n}\n.dashboard-container .graph .wrapper {\n      height: 320px;\n}\n.dashboard-container .graph .wrapper .echarts {\n        width: 100%;\n        height: 100%;\n}\n.dashboard-container .log-wrapper {\n    margin: 24px 0 16px;\n    padding: 0 24px 24px;\n    background: #fff;\n    border-radius: 4px;\n}\n.dashboard-container .log-wrapper .title {\n      padding-top: 20px;\n      margin-bottom: 15px;\n      font-size: 18px;\n      color: #534c77;\n}\n.dashboard-container .log-wrapper .tab-bar {\n      margin-left: 18px;\n}\n.dashboard-container .log-wrapper .tab-bar .tab-btn {\n        display: -webkit-box;\n        display: -ms-flexbox;\n        display: flex;\n        -webkit-box-align: center;\n            -ms-flex-align: center;\n                align-items: center;\n        margin-right: 24px;\n        padding: 0 5px;\n        background: #f8f8fa;\n        line-height: 26px;\n        border-radius: 26px;\n        cursor: pointer;\n}\n.dashboard-container .log-wrapper .tab-bar .tab-btn .text {\n          padding: 0 10px;\n          font-size: 16px;\n          font-weight: bold;\n          color: #7f7d8e;\n}\n.dashboard-container .log-wrapper .tab-bar .tab-btn .count {\n          min-width: 16px;\n          height: 16px;\n          padding: 0 5px;\n          border-radius: 16px;\n          line-height: 16px;\n          text-align: center;\n          color: #fff;\n}\n.dashboard-container .log-wrapper .tab-bar .tab-btn .error {\n          background: #ff6464;\n}\n.dashboard-container .log-wrapper .tab-bar .tab-btn .warning {\n          background: #ff5d93;\n}\n.dashboard-container .log-wrapper .tab-bar .tab-btn .info {\n          background: #ffd70d;\n}\n.dashboard-container .log-wrapper .tab-bar .tab-btn .debug {\n          background: #24b68b;\n}\n.dashboard-container .log-wrapper .tab-bar .tab-btn:hover {\n          background: #494ece;\n}\n.dashboard-container .log-wrapper .tab-bar .tab-btn:hover .text {\n            color: #fff;\n}\n.dashboard-container .log-wrapper .tab-bar .tab-btn-active {\n        background: #494ece;\n}\n.dashboard-container .log-wrapper .tab-bar .tab-btn-active .text {\n          color: #fff;\n}\n.dashboard-container .log-wrapper .log-container {\n      height: 280px;\n      padding: 24px;\n      background: #f8f8fa;\n      overflow: auto;\n}\n.dashboard-container .log-wrapper .log-container .log-list > li {\n        /*height: 25px;*/\n        line-height: 25px;\n        text-indent: 10px;\n}\n",""])},zH3H:function(t,a,n){(t.exports=n("I1BE")(!1)).push([t.i,"",""])}}]);