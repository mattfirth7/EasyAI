const fs = require('fs');
const AWS = require('aws-sdk');
const express = require('express');
const fileUpload = require('express-fileupload');
const cors = require('cors');
const bodyParser = require('body-parser');
const morgan = require('morgan');
const _ = require('lodash');
var spawn = require('child_process').spawn;
const s3 = new AWS.S3();

const getModelsList = () => {
	var params = {
		Bucket: 'graymatter-file-storage',
		Prefix: 'models'
	};
	s3.listObjectsV2(params, function(err, data) {
		if (err) {
			console.log(err, err.stack);
		}
		else  {
			var modelsList = [];
			data.Contents.forEach(item => {
				modelsList.push(item.Key);
			});
			return modelsList;
		}
	});
};

app.get('/predict', function(req, res) {
	var modelsList = getModelsList();
	console.log(modelsList);
	res.render('predict', {page:'Predict', menuId:'predict', models:modelsList});
});