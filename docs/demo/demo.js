let META = {};

fetch('data/meta.json')
    .then(r => r.json())
    .then(data => { 
        META = data; 

        const dateSelect = document.getElementById('dateSelect');
        dateSelect.innerHTML = '';
        Object.entries(META).forEach(([key, val]) => {
            if (val.label) { 
                const el = document.createElement('option');
                el.value = key;
                el.textContent = val.label;
                dateSelect.appendChild(el);
            }
        });
        populateSelect('regionSelect', data.regions);
        loadDate();
    });
 
function populateSelect(id, options) {
    const select = document.getElementById(id);
    select.innerHTML = '';
    options.forEach(opt => {
        const el = document.createElement('option');
        el.value = opt.value;
        el.textContent = opt.label;
        select.appendChild(el);
    });
}    
    
let running = false;

function getImagePaths(date, region) {
    const base = `data/${date}_${region}_`;
    return {
        era5: base + 'era5.png',
        ngcd: base + 'ngcd.png',
        pred: base + 'pred.png'
    };
}

function loadDate() {
    const key = document.getElementById('dateSelect').value;
    const reg = document.getElementById('regionSelect').value;
    const imgs = getImagePaths(key, reg)

    document.getElementById('imgERA5').src = imgs["era5"]
    document.getElementById('imgNGCD').src = imgs["ngcd"]
    const predim = document.getElementById('imgPred')
    predim.src = imgs["pred"]
    predim.style.opacity = '0.2';
    predim.style.filter = 'blur(4px)';

    document.getElementById('infoRow').innerHTML = `Press <strong>Run model</strong> for the prediction.`;
    document.getElementById('progressFill').style.width = '0%';
    document.getElementById('mae').innerHTML = `&mdash;`

    const msg = document.getElementById('msg');
    msg.style.display = 'block'
}

function runModel() {
    if (running) return;
    running = true;

    const btn = document.getElementById('runBtn');
    const fill = document.getElementById('progressFill');
    const key = document.getElementById('dateSelect').value;
    const d = META[key];
    
    btn.textContent = 'Running...';
    fill.style.width = '0%';

    const steps = [
        { percent: 20, delay: 200, label: 'Processing patches...' },
        { percent: 50, delay: 600, label: 'Flow matching t=0 &rarr; 0.5...' },
        { percent: 80, delay: 600, label: 'Flow matching t=0.5 &rarr; 1.0...' },
        { percent: 100, delay: 400, label: 'Model predition complete.' },
    ];

    let i = 0;

    function nextStep() {
        if (i >= steps.length) {
            const msg = document.getElementById('msg');
            const predim = document.getElementById('imgPred');

            msg.style.display = 'none'
            predim.style.opacity = '1';
            predim.style.filter = 'none';

            document.getElementById('mae').innerHTML = `MAE: <span class="mae-badge">${d.mae}</span> (input: ${d.input_mae})`;
            document.getElementById('infoRow').innerHTML = `${steps[steps.length -1].label} Temperature range: ${d.temp_min} &ndash; ${d.temp_max}`
            btn.innerHTML = 'Run model';
            running = false;
            return;
        }
        const step = steps[i];
        fill.style.width = step.percent + '%';
        document.getElementById('infoRow').innerHTML = step.label
        i++;
        setTimeout(nextStep, step.delay);
    }
    nextStep();
}
document.getElementById('dateSelect').addEventListener('change', loadDate);
document.getElementById('regionSelect').addEventListener('change', loadDate);