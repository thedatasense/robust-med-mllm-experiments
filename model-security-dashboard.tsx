import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, LineChart, Line, 
  ScatterChart, Scatter, ZAxis, ComposedChart, Cell, Area } from 'recharts';

const Dashboard = () => {
  const [activeView, setActiveView] = useState('overview');

  const modelColors = {
    'Llama-3.2-11B': '#8884d8',
    'Gemma-3-4b': '#82ca9d',
    'GPT-4o': '#ffc658',
    'Llava-Med-7b': '#ff8042',
    'CheXagent-8b': '#0088fe'
  };

  const attackCategoryColors = {
    'Persistence of Attack Effects': '#8884d8',
    'Prompt Injection Effectiveness': '#82ca9d',
    'Safety Bypass Success': '#ffc658',
    'Text Jailbreak Success': '#ff8042',
    'Impact on Medical Decision Support': '#0088fe',
    'Confidentiality Breach': '#8dd1e1',
    'Misinformation or Malicious Content Generation': '#a4de6c',
    'Visual Jailbreak Success': '#d0ed57',
    'Denial-of-Service Attack Resilience': '#83a6ed'
  };

  const overallVulnerabilityData = [
    { name: 'Llama-3.2-11B', score: 0.75, maxVulnerability: 1.19, stdDev: 0.37 },
    { name: 'Gemma-3-4b', score: 0.70, maxVulnerability: 1.29, stdDev: 0.36 },
    { name: 'GPT-4o', score: 0.32, maxVulnerability: 0.69, stdDev: 0.28 },
    { name: 'Llava-Med-7b', score: 0.29, maxVulnerability: 0.84, stdDev: 0.30 },
    { name: 'CheXagent-8b', score: 0.18, maxVulnerability: 0.68, stdDev: 0.28 }
  ];

  const attackCategoryEffectiveness = [
    { name: 'Persistence of Attack Effects', score: 0.899490 },
    { name: 'Prompt Injection Effectiveness', score: 0.739951 },
    { name: 'Safety Bypass Success', score: 0.630739 },
    { name: 'Text Jailbreak Success', score: 0.473495 },
    { name: 'Impact on Medical Decision Support', score: 0.455426 },
    { name: 'Confidentiality Breach', score: 0.412415 },
    { name: 'Misinformation or Malicious Content Generation', score: 0.308546 },
    { name: 'Visual Jailbreak Success', score: 0.064367 },
    { name: 'Denial-of-Service Attack Resilience', score: 0.032786 }
  ];

  const detailedVulnerabilityData = [
    {
      model: 'Llama-3.2-11B',
      'Confidentiality Breach': 0.64,
      'Denial-of-Service Attack Resilience': 0.288,
      'Impact on Medical Decision Support': 0.76,
      'Misinformation or Malicious Content Generation': 0.53,
      'Persistence of Attack Effects': 1.18,
      'Prompt Injection Effectiveness': 1.19,
      'Safety Bypass Success': 1.16,
      'Text Jailbreak Success': 0.85,
      'Visual Jailbreak Success': 0.12
    },
    {
      model: 'Gemma-3-4b',
      'Confidentiality Breach': 0.88,
      'Denial-of-Service Attack Resilience': 0.293,
      'Impact on Medical Decision Support': 0.65,
      'Misinformation or Malicious Content Generation': 0.58,
      'Persistence of Attack Effects': 1.29,
      'Prompt Injection Effectiveness': 0.97,
      'Safety Bypass Success': 0.78,
      'Text Jailbreak Success': 0.78,
      'Visual Jailbreak Success': 0.09
    },
    {
      model: 'GPT-4o',
      'Confidentiality Breach': 0.22,
      'Denial-of-Service Attack Resilience': -0.23,
      'Impact on Medical Decision Support': 0.55,
      'Misinformation or Malicious Content Generation': 0.22,
      'Persistence of Attack Effects': 0.69,
      'Prompt Injection Effectiveness': 0.28,
      'Safety Bypass Success': 0.66,
      'Text Jailbreak Success': 0.33,
      'Visual Jailbreak Success': 0.13
    },
    {
      model: 'Llava-Med-7b',
      'Confidentiality Breach': 0.20,
      'Denial-of-Service Attack Resilience': 0.00008,
      'Impact on Medical Decision Support': 0.28,
      'Misinformation or Malicious Content Generation': 0.17,
      'Persistence of Attack Effects': 0.84,
      'Prompt Injection Effectiveness': 0.58,
      'Safety Bypass Success': 0.30,
      'Text Jailbreak Success': 0.26,
      'Visual Jailbreak Success': -0.03
    },
    {
      model: 'CheXagent-8b',
      'Confidentiality Breach': 0.12,
      'Denial-of-Service Attack Resilience': -0.187,
      'Impact on Medical Decision Support': 0.05,
      'Misinformation or Malicious Content Generation': 0.03,
      'Persistence of Attack Effects': 0.49,
      'Prompt Injection Effectiveness': 0.68,
      'Safety Bypass Success': 0.26,
      'Text Jailbreak Success': 0.15,
      'Visual Jailbreak Success': 0.01
    }
  ];

  const radarData = detailedVulnerabilityData.map(item => {
    const newItem = { model: item.model };
    Object.keys(item).forEach(key => {
      if (key !== 'model') {
        newItem[key.split(' ').join('')] = item[key];
      }
    });
    return newItem;
  });

  const prioritizationData = [
    { name: 'Persistence of Attack Effects', effectiveness: 0.90, prevalence: 1.0, priority: 'Critical' },
    { name: 'Prompt Injection Effectiveness', effectiveness: 0.74, prevalence: 1.0, priority: 'Critical' },
    { name: 'Safety Bypass Success', effectiveness: 0.63, prevalence: 1.0, priority: 'Critical' },
    { name: 'Text Jailbreak Success', effectiveness: 0.47, prevalence: 0.8, priority: 'High' },
    { name: 'Impact on Medical Decision Support', effectiveness: 0.46, prevalence: 0.8, priority: 'High' },
    { name: 'Confidentiality Breach', effectiveness: 0.41, prevalence: 0.4, priority: 'Medium' },
    { name: 'Misinformation or Malicious Content Generation', effectiveness: 0.31, prevalence: 0.4, priority: 'Medium' },
    { name: 'Denial-of-Service Attack Resilience', effectiveness: 0.03, prevalence: 0.4, priority: 'Medium' },
    { name: 'Visual Jailbreak Success', effectiveness: 0.06, prevalence: 0.0, priority: 'Low' }
  ];

  const modelTypeData = [
    { type: 'General-Purpose', vulnerability: 0.59 },
    { type: 'Medical-Specialized', vulnerability: 0.23 },
    { type: 'With Vision', vulnerability: 0.38 },
    { type: 'Text-Only', vulnerability: 0.70 },
    { type: 'Large Models', vulnerability: 0.32 },
    { type: 'Medium Models', vulnerability: 0.75 },
    { type: 'Small Models', vulnerability: 0.39 }
  ];

  const evaluatorAgreementData = [
    { name: 'Visual Jailbreak Success', correlation: 0.7644, mae: 3.0625 },
    { name: 'Impact on Medical Decision Support', correlation: 0.7062, mae: 3.2972 },
    { name: 'Misinformation or Malicious Content Generation', correlation: 0.6747, mae: 3.9493 },
    { name: 'Confidentiality Breach', correlation: 0.5267, mae: 5.0195 },
    { name: 'Prompt Injection Effectiveness', correlation: 0.5218, mae: 4.4369 },
    { name: 'Denial-of-Service Attack Resilience', correlation: 0.4175, mae: 3.9109 },
    { name: 'Safety Bypass Success', correlation: 0.3976, mae: 4.6273 },
    { name: 'Text Jailbreak Success', correlation: 0.3204, mae: 5.2246 },
    { name: 'Persistence of Attack Effects', correlation: 0.2171, mae: 6.3170 }
  ];

  const renderOverview = () => (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">Overall Model Vulnerability</h2>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={overallVulnerabilityData}
            margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" angle={-45} textAnchor="end" />
            <YAxis label={{ value: 'Vulnerability Score (Z-score)', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Bar dataKey="score" name="Average Vulnerability" fill="#8884d8" />
            <Bar dataKey="maxVulnerability" name="Maximum Vulnerability" fill="#82ca9d" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <h2 className="text-xl font-bold mt-8 mb-4">Attack Category Effectiveness</h2>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={attackCategoryEffectiveness}
            margin={{ top: 20, right: 30, left: 20, bottom: 100 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" angle={-45} textAnchor="end" />
            <YAxis label={{ value: 'Effectiveness Score (Z-score)', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Bar dataKey="score" name="Average Effectiveness" fill="#82ca9d">
              {
                attackCategoryEffectiveness.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={Object.values(attackCategoryColors)[index % Object.values(attackCategoryColors).length]} />
                ))
              }
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  const renderDetailedAnalysis = () => (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">Model Comparison by Attack Category</h2>
      <div className="h-96">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart outerRadius={150} width={730} height={400} data={radarData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="model" />
            <PolarRadiusAxis angle={30} domain={[0, 1.5]} />
            <Radar name="Llama-3.2-11B" dataKey="ConfidentialityBreach" stroke={modelColors['Llama-3.2-11B']} fill={modelColors['Llama-3.2-11B']} fillOpacity={0.6} />
            <Radar name="Gemma-3-4b" dataKey="Denial-of-ServiceAttackResilience" stroke={modelColors['Gemma-3-4b']} fill={modelColors['Gemma-3-4b']} fillOpacity={0.6} />
            <Radar name="GPT-4o" dataKey="ImpactonMedicalDecisionSupport" stroke={modelColors['GPT-4o']} fill={modelColors['GPT-4o']} fillOpacity={0.6} />
            <Radar name="Llava-Med-7b" dataKey="MisinformationorMaliciousContentGeneration" stroke={modelColors['Llava-Med-7b']} fill={modelColors['Llava-Med-7b']} fillOpacity={0.6} />
            <Radar name="CheXagent-8b" dataKey="PersistenceofAttackEffects" stroke={modelColors['CheXagent-8b']} fill={modelColors['CheXagent-8b']} fillOpacity={0.6} />
            <Legend />
            <Tooltip />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      <h2 className="text-xl font-bold mt-8 mb-4">Attack Category Prioritization</h2>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart
            margin={{ top: 20, right: 30, left: 20, bottom: 80 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="effectiveness" 
              type="number" 
              name="Effectiveness" 
              label={{ value: 'Effectiveness', position: 'bottom', offset: 0 }} 
              domain={[0, 1]}
            />
            <YAxis 
              dataKey="prevalence" 
              type="number" 
              name="Prevalence" 
              label={{ value: 'Prevalence', angle: -90, position: 'insideLeft' }}
              domain={[0, 1.1]}
            />
            <ZAxis 
              dataKey="name" 
              name="Attack Category" 
              range={[50, 500]} 
            />
            <Tooltip 
              cursor={{ strokeDasharray: '3 3' }} 
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="bg-white p-2 border border-gray-300 rounded">
                      <p>{`${payload[0].payload.name}`}</p>
                      <p>{`Effectiveness: ${payload[0].value.toFixed(2)}`}</p>
                      <p>{`Prevalence: ${payload[1].value.toFixed(2)}`}</p>
                      <p>{`Priority: ${payload[0].payload.priority}`}</p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Legend />
            <Scatter 
              name="Attack Categories" 
              data={prioritizationData} 
              fill="#8884d8"
            >
              {
                prioritizationData.map((entry, index) => {
                  let color;
                  switch(entry.priority) {
                    case 'Critical': color = '#ff0000'; break;
                    case 'High': color = '#ff8c00'; break;
                    case 'Medium': color = '#ffff00'; break;
                    case 'Low': color = '#00ff00'; break;
                    default: color = '#8884d8';
                  }
                  return <Cell key={`cell-${index}`} fill={color} />;
                })
              }
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  const renderEvaluatorAgreement = () => (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">Evaluator Agreement by Attack Category</h2>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={evaluatorAgreementData}
            margin={{ top: 20, right: 30, left: 20, bottom: 100 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" angle={-45} textAnchor="end" />
            <YAxis yAxisId="left" label={{ value: 'Correlation', angle: -90, position: 'insideLeft' }} />
            <YAxis yAxisId="right" orientation="right" label={{ value: 'MAE', angle: 90, position: 'insideRight' }} />
            <Tooltip />
            <Legend />
            <Bar yAxisId="left" dataKey="correlation" name="Correlation" fill="#8884d8" />
            <Line yAxisId="right" type="monotone" dataKey="mae" name="Mean Absolute Error" stroke="#ff7300" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <h2 className="text-xl font-bold mt-8 mb-4">Vulnerability by Model Type</h2>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={modelTypeData}
            margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="type" angle={-45} textAnchor="end" />
            <YAxis domain={[0, 0.8]} label={{ value: 'Average Vulnerability Score', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Bar dataKey="vulnerability" name="Vulnerability Score" fill="#8884d8">
              {
                modelTypeData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={index < 2 ? '#82ca9d' : index < 4 ? '#8884d8' : '#ffc658'} />
                ))
              }
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  const renderModelProfiles = () => (
    <div className="p-4">
      {detailedVulnerabilityData.map((model, idx) => (
        <div key={idx} className="mb-10 pb-6 border-b border-gray-200">
          <h2 className="text-xl font-bold mb-4">{model.model} Vulnerability Profile</h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={Object.entries(model)
                  .filter(([key]) => key !== 'model')
                  .map(([key, value]) => ({ name: key, score: value }))}
                margin={{ top: 20, right: 30, left: 20, bottom: 80 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" angle={-45} textAnchor="end" />
                <YAxis domain={[-0.3, 1.5]} label={{ value: 'Vulnerability Score (Z-score)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Bar dataKey="score" name="Vulnerability Score" fill={modelColors[model.model]}>
                  {
                    Object.entries(model)
                      .filter(([key]) => key !== 'model')
                      .map(([key], index) => {
                        let color;
                        const value = model[key];
                        if (value >= 0.8) color = '#ff0000'; // High
                        else if (value >= 0.4) color = '#ff8c00'; // Moderate
                        else if (value > 0) color = '#ffff00'; // Low
                        else color = '#00ff00'; // Resistant
                        return <Cell key={`cell-${index}`} fill={color} />;
                      })
                  }
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          
          <div className="mt-4">
            <h3 className="text-lg font-semibold mb-2">Key Vulnerabilities:</h3>
            <ul className="list-disc pl-6">
              {Object.entries(model)
                .filter(([key, value]) => key !== 'model' && value >= 0.4)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 3)
                .map(([key, value], idx) => (
                  <li key={idx} className="mb-1">
                    <span className="font-medium">{key}:</span> {value.toFixed(2)} 
                    ({value >= 0.8 ? 'High' : 'Moderate'} Vulnerability)
                  </li>
                ))
              }
            </ul>
          </div>
          
          <div className="mt-4">
            <h3 className="text-lg font-semibold mb-2">Strengths:</h3>
            <ul className="list-disc pl-6">
              {Object.entries(model)
                .filter(([key, value]) => key !== 'model' && value <= 0.1)
                .sort((a, b) => a[1] - b[1])
                .slice(0, 2)
                .map(([key, value], idx) => (
                  <li key={idx} className="mb-1">
                    <span className="font-medium">{key}:</span> {value.toFixed(2)} 
                    ({value <= 0 ? 'Resistant' : 'Low Vulnerability'})
                  </li>
                ))
              }
            </ul>
          </div>
        </div>
      ))}
    </div>
  );

  const renderContent = () => {
    switch(activeView) {
      case 'overview':
        return renderOverview();
      case 'detailed':
        return renderDetailedAnalysis();
      case 'evaluator':
        return renderEvaluatorAgreement();
      case 'profiles':
        return renderModelProfiles();
      default:
        return renderOverview();
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto py-4 px-4">
          <h1 className="text-2xl font-bold text-gray-900">AI Model Security Vulnerability Dashboard</h1>
          <p className="text-sm text-gray-600 mt-1">Analysis of 5 models across 9 attack categories</p>
        </div>
      </header>
      
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex space-x-4">
            <button 
              onClick={() => setActiveView('overview')}
              className={`px-3 py-2 text-sm font-medium ${activeView === 'overview' ? 'text-blue-700 border-b-2 border-blue-700' : 'text-gray-500 hover:text-gray-700'}`}
            >
              Overview
            </button>
            <button 
              onClick={() => setActiveView('detailed')}
              className={`px-3 py-2 text-sm font-medium ${activeView === 'detailed' ? 'text-blue-700 border-b-2 border-blue-700' : 'text-gray-500 hover:text-gray-700'}`}
            >
              Detailed Analysis
            </button>
            <button 
              onClick={() => setActiveView('evaluator')}
              className={`px-3 py-2 text-sm font-medium ${activeView === 'evaluator' ? 'text-blue-700 border-b-2 border-blue-700' : 'text-gray-500 hover:text-gray-700'}`}
            >
              Evaluator Agreement
            </button>
            <button 
              onClick={() => setActiveView('profiles')}
              className={`px-3 py-2 text-sm font-medium ${activeView === 'profiles' ? 'text-blue-700 border-b-2 border-blue-700' : 'text-gray-500 hover:text-gray-700'}`}
            >
              Model Profiles
            </button>
          </div>
        </div>
      </nav>
      
      <main className="max-w-7xl mx-auto py-6 px-4">
        <div className="bg-white shadow rounded-lg">
          {renderContent()}
        </div>
        
        <div className="bg-white shadow rounded-lg mt-6 p-4">
          <h2 className="text-xl font-bold mb-4">Key Findings</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="border rounded p-3 bg-blue-50">
              <h3 className="font-medium text-lg">Most Effective Attack Categories</h3>
              <ol className="list-decimal pl-5 mt-2">
                <li>Persistence of Attack Effects (0.90)</li>
                <li>Prompt Injection Effectiveness (0.74)</li>
                <li>Safety Bypass Success (0.63)</li>
              </ol>
            </div>
            
            <div className="border rounded p-3 bg-green-50">
              <h3 className="font-medium text-lg">Most Secure Models</h3>
              <ol className="list-decimal pl-5 mt-2">
                <li>CheXagent-8b (0.18)</li>
                <li>Llava-Med-7b (0.29)</li>
                <li>GPT-4o (0.32)</li>
              </ol>
            </div>
            
            <div className="border rounded p-3 bg-red-50">
              <h3 className="font-medium text-lg">Most Vulnerable Models</h3>
              <ol className="list-decimal pl-5 mt-2">
                <li>Llama-3.2-11B (0.75)</li>
                <li>Gemma-3-4b (0.70)</li>
              </ol>
            </div>
            
            <div className="border rounded p-3 bg-yellow-50">
              <h3 className="font-medium text-lg">Model Type Insights</h3>
              <ul className="list-disc pl-5 mt-2">
                <li>Medical models are more secure (0.23) than general-purpose models (0.59)</li>
                <li>Models with vision capabilities are more secure (0.38) than text-only models (0.70)</li>
              </ul>
            </div>
          </div>
        </div>
        
        <div className="bg-white shadow rounded-lg mt-6 p-4">
          <h2 className="text-xl font-bold mb-4">Recommendations</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="border rounded p-3">
              <h3 className="font-medium text-lg">Critical Priority</h3>
              <ul className="list-disc pl-5 mt-2">
                <li>Improve context reset mechanisms to address Persistence of Attack Effects</li>
                <li>Implement prompt boundary enforcement for Prompt Injection</li>
                <li>Add multi-layer safety checks for Safety Bypass attempts</li>
              </ul>
            </div>
            
            <div className="border rounded p-3">
              <h3 className="font-medium text-lg">By Model Type</h3>
              <ul className="list-disc pl-5 mt-2">
                <li><span className="font-medium">General-Purpose:</span> Focus on persistence and safety bypass</li>
                <li><span className="font-medium">Medical:</span> Focus on persistence and prompt injection</li>
                <li><span className="font-medium">Vision-Capable:</span> Focus on persistence and prompt injection</li>
              </ul>
            </div>
            
            <div className="border rounded p-3">
              <h3 className="font-medium text-lg">Implementation Strategy</h3>
              <ol className="list-decimal pl-5 mt-2">
                <li>Prioritize critical attack categories</li>
                <li>Apply model-specific improvements</li>
                <li>Implement general best practices</li>
                <li>Establish ongoing monitoring</li>
              </ol>
            </div>
          </div>
        </div>
      </main>
      
      <footer className="bg-white mt-6 py-4 border-t">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm text-gray-500">
          Dataset: 68,478 evaluations across 369 unique questions and 27 question types
        </div>
      </footer>
    </div>
  );
};

export default Dashboard;
