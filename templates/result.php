<!DOCTYPE html>

<html>

<head>
  <style>
    .container {
      background-color: lightgoldenrodyellow;
      padding: 30px;
      text-align: center;
      margin: 0px;
    }

    body {
      background-color: lightsalmon;
      margin: 30px;
      font-size: larger;
    }

    h1, h2 {
      text-align: center;
    }

    .custom-back {
      border: 2px solid black;
      display: inline-block;
      padding: 10px 20px;
      border-radius: 10px;
      cursor: pointer;
      background-color: lightblue;
      color: black;
      
    }
  </style>
</head>

<body>
  <h1>Weather Classification</h1>
  <div class="container">
    <h3>Your Image Classified As</h2>
    <h2>{{ss}}</h2>
    <a href="http://127.0.0.1:5000/"><span class="custom-back">Back</span></a>
  </div>
</body>

</html>