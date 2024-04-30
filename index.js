
async function runExample() {

    var x = new Float32Array( 1, 8 )

    var x = [];

     x[0] = document.getElementById('box1').value;
     x[1] = document.getElementById('box2').value;
     x[2] = document.getElementById('box3').value;
     x[3] = document.getElementById('box4').value;
     x[4] = document.getElementById('box5').value;
     x[5] = document.getElementById('box6').value;
     x[6] = document.getElementById('box7').value;
     x[7] = document.getElementById('box8').value;
     //x[8] = document.getElementById('box9').value;
     

    let tensorX = new onnx.Tensor(x, 'float32', [1, 8]);
    let feeds = {float_input: tensorX};

    let session = await ort.InferenceSession.create('onnx_model .onnx');
    let result = await session.run(feeds);
    let outputData = result.variable.data;
    
    outputData = parseFloat(outputData).toFixed(2)
   let predictions = document.getElementById('predictions');

  predictions.innerHTML = ` <hr> Got an output tensor with values: <br/>
   <table>
     <tr>
       <td>  Concrete Result:  </td>
       <td id="td0">  ${outputData.data[0].toFixed(2)}  </td> //may have to change this
     </tr>
  </table>`;
    


}
