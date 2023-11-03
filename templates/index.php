<!DOCTYPE html>

<html>

<?php
$folder_path = "uploaded";

$files = glob($folder_path . '/*');

foreach ($files as $file) {

  if (is_file($file))

    unlink($file);
}
?>

<head>
  <link class="jsbin" href="http://ajax.googleapis.com/ajax/libs/jqueryui/1/themes/base/jquery-ui.css" rel="stylesheet" type="text/css" />
  <script class="jsbin" src="http://ajax.googleapis.com/ajax/libs/jquery/1/jquery.min.js"></script>
  <script class="jsbin" src="http://ajax.googleapis.com/ajax/libs/jqueryui/1.8.0/jquery-ui.min.js"></script>
  <meta charset=utf-8 />
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

    h1 {
      text-align: center;
    }

    button {
      background-color: lightgreen;
      padding-top: 10px;
      padding-bottom: 10px;
      padding-left: 50px;
      padding-right: 50px;
      font-size: large;
      font-weight: bold;
      border-radius: 10px;
      cursor: pointer;
    }

    input[type="file"] {
      display: none;
    }

    .custom-file-upload {
      border: 2px solid black;
      display: inline-block;
      padding: 10px 20px;
      border-radius: 10px;
      cursor: pointer;
      background-color: lightblue;
    }
  </style>
</head>

<body>
  <h1>Weather Classification</h1>
  <div class="container">
    <form action="http://127.0.0.1:5000/uploader" method="POST" enctype="multipart/form-data">
      <label for="file" class="custom-file-upload">
        Upload Your Image
      </label>
      <input type='file' id="file" name='file' onchange="readURL(this);" required /><br><br>
      <img style="max-width: 700px; max-height: 700px" id="blah" src="#" /><br><br>
      <button type="submit">Predict</button>
    </form>
  </div>
</body>

<script>
  function readURL(input) {
    if (input.files && input.files[0]) {
      var reader = new FileReader();

      reader.onload = function(e) {
        $('#blah').attr('src', e.target.result);
      };

      reader.readAsDataURL(input.files[0]);
    }
  }
</script>

</html>