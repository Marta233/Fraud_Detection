import React, { useState } from 'react';
import axios from 'axios';
import './styles.css';

const FraudForm = () => {
    const [formData, setFormData] = useState({
        purchase_time: '',
        signup_time: '',
        source: '',
        browser: '',
        sex: '',
        purchase_value: '',
        age: '',
        country: ''
    });

    const [result, setResult] = useState('');
    const [resultColor, setResultColor] = useState('black');
    const [loading, setLoading] = useState(false);

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            const response = await axios.post('http://127.0.0.1:5000/predict', formData);
            const { message, status } = response.data;

            // Set the result message and color based on status
            setResult(message);
            setResultColor(status === 1 ? 'green' : 'red');
        } catch (error) {
            console.error('There was an error!', error);
            setResult('Error: Could not get prediction.');
            setResultColor('black');
        }
        setLoading(false);
    };

    return (
        <div className="background-image">
            <h2>Fraud Detection Form</h2>
            <div className="form-container">
                {result && (
                    <h3 className="result-message" style={{ color: resultColor }}>
                        {result}
                    </h3>
                )}
                
                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label>Purchase Time</label>
                        <input
                            type="datetime-local"
                            name="purchase_time"
                            value={formData.purchase_time}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Signup Time</label>
                        <input
                            type="datetime-local"
                            name="signup_time"
                            value={formData.signup_time}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Source</label>
                        <select
                            name="source"
                            value={formData.source}
                            onChange={handleChange}
                            required
                        >
                            <option value="" disabled>Select Source</option>
                            <option value="Ads">Ads</option>
                            <option value="Direct">Direct</option>
                            <option value="SEO">SEO</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label>Browser</label>
                        <select
                            name="browser"
                            value={formData.browser}
                            onChange={handleChange}
                            required
                        >
                            <option value="" disabled>Select Browser</option>
                            <option value="Chrome">Chrome</option>
                            <option value="Firefox">Firefox</option>
                            <option value="Safari">Safari</option>
                            <option value="Opera">Opera</option>
                            <option value="Edge">Edge</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label>Gender</label>
                        <select
                            name="sex"
                            value={formData.sex}
                            onChange={handleChange}
                            required
                        >
                            <option value="" disabled>Select Gender</option>
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label>Purchase Value</label>
                        <input
                            type="number"
                            name="purchase_value"
                            value={formData.purchase_value}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Age</label>
                        <input
                            type="number"
                            name="age"
                            value={formData.age}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Country</label>
                        <input
                            type="text"
                            name="country"
                            value={formData.country}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <button type="submit" disabled={loading}>
                        {loading ? 'Loading...' : 'Submit'}
                    </button>
                </form>
            </div>
        </div>
    );
};

export default FraudForm;
