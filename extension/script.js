import config from './config.json' assert { type: 'json' };
import generationConfig from './generation_config.json' assert { type: 'json' };
console.log('Generation config: ' + JSON.stringify(generationConfig));
const headers = {'Authorization': 'Bearer ' + config.huggingface_token};

document.getElementById('submit').addEventListener('click', async function (event) {
    var text = document.getElementById('text').value;
    let simplification = '';
    simplification = await simplifyText(text).then((response) => {console.log(response); return response});
    console.log(text + ' -> ' + simplification);
    document.getElementById('simplifiedText').innerHTML = simplification;
});

async function simplifyText(text){
    let simplifiedSents = [];
    const sents = text.split('.');
    for (let i = 0; i < sents.length; i++){
        var simplifiedSent = await simplifySent(sents[i]);
        simplifiedSents.push(simplifiedSent); 
    }
    return simplifiedSents.join('. ')
};

async function simplifySent(text){
    const params = {
        'do_sample':config.do_sample,
        'repetition_penalty':config.repetition_penalty,
        'max_length':text.split(' ').length * config.max_length_factor
    };
    const response = await fetch(
		config.api_url,
		{
			headers: headers,
			method: 'POST',
			body: JSON.stringify({'inputs': text, 'parameters':params})
		}
	);
	const result = await response.json();
    console.log(result);
    if (result & result[0]){
        return result[0].generated_text
    }
    return text
};
