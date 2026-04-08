function d3graphscript(config = {
    // Default values
    width: 800,
    height: 600,
    charge: -250,
    distance: 0,
    directed: false,
    collision: 0.5,
    link_tension: 1,
    sticky: false,
    background_color: '#FFFFFF',
    node_text_inside: false,
    }) {
    
    //Constants for the SVG
    var width = config.width;
    var height = config.height;
    var background_color = config.background_color || '#FFFFFF';
    var sticky = config.sticky || false;
    
    // Set the body background color
    document.body.style.backgroundColor = background_color;
    
    //Set up the colour scale
    var color = d3.scale.category20();
    
    var force = d3.layout.force()
      .charge(config.charge)
      .linkDistance((d) => d.edge_distance || config.distance)
      //.linkDistance((d) => config.distance > 0 ? config.distance : d.edge_weight)
      .linkStrength(config.link_tension !== undefined ? config.link_tension : 1)
      .size([width, height]);
    
    // ---- DRAGGING ----
    // Sticky mode: dragstart fixes the node so the simulation stops pulling it.
    //   dragend keeps it pinned (dashed stroke indicator).
    //   Right-click a pinned node to release it back into the simulation.
    // Normal mode: standard free-drag behaviour is preserved.
    
    function dragstarted(d) {
      d3.event.sourceEvent.stopPropagation();
      d3.select(this).classed("dragging", true);
      if (sticky) {
        d.fixed = true;
        force.start();
      }
    }
    
    function dragged(d) {
      if (sticky) {
        d.x = d.px = d3.event.x;
        d.y = d.py = d3.event.y;
      } else {
        d3.select(this).attr("cx", d.x = d3.event.x).attr("cy", d.y = d3.event.y);
      }
    }
    
    function dragended(d) {
      d3.select(this).classed("dragging", false);
      if (sticky) {
        // Keep the node fixed and apply a visual "pinned" cue (dashed border)
        d.fixed = true;
        d3.select(this).select(".node-shape")
          .style("stroke-dasharray", "4,2")
          .style("stroke-width", function(d) { return Math.max(parseFloat(d.node_size_edge) || 1, 2); });
      }
    }
    
    var drag = force.drag()
      .origin(function(d) { return d; })
      .on("dragstart", dragstarted)
      .on("drag", dragged)
      .on("dragend", dragended);
    
    // ---- END DRAGGING ----

    // =====================================================================
    // ---- SHAPE RENDERING ------------------------------------------------
    // Supported node_marker values:
    //   'circle'                        – SVG <circle>  (default)
    //   'ellipse'                       – SVG <ellipse> (wider than tall)
    //   'square'                        – square via <path>
    //   'rectangle'                     – wide rectangle via <path>
    //   'triangle' / 'triangle-up'      – upward equilateral triangle
    //   'triangle-down'                 – downward triangle
    //   'diamond'                       – rotated square
    //   'star'                          – 5-pointed star
    //   'hexagon'                       – regular hexagon
    //   'pentagon'                      – regular pentagon
    // =====================================================================

    /**
     * Returns an SVG path `d` string for polygon/star shapes, centred at (0,0).
     * Returns null for 'circle' and 'ellipse' (handled as native SVG elements).
     */
    function shapePathD(marker, r) {
      r = parseFloat(r) || 8;
      var m = (marker || 'circle').toLowerCase();
      switch (m) {

        case 'triangle':
        case 'triangle-up': {
          var h = r * Math.sqrt(3);
          return 'M 0,' + (-r) +
                 ' L ' + (h / 2) + ',' + (r / 2) +
                 ' L ' + (-h / 2) + ',' + (r / 2) + ' Z';
        }

        case 'triangle-down': {
          var h = r * Math.sqrt(3);
          return 'M 0,' + r +
                 ' L ' + (h / 2) + ',' + (-r / 2) +
                 ' L ' + (-h / 2) + ',' + (-r / 2) + ' Z';
        }

        case 'square': {
          var s = r * 1.4;
          return 'M ' + (-s) + ',' + (-s) +
                 ' L ' +   s + ',' + (-s) +
                 ' L ' +   s + ',' +   s  +
                 ' L ' + (-s) + ',' +   s  + ' Z';
        }

        case 'rect':
        case 'rectangle': {
          var w = r * 2.0, h = r * 1.0;
          return 'M ' + (-w) + ',' + (-h) +
                 ' L ' +   w + ',' + (-h) +
                 ' L ' +   w + ',' +   h  +
                 ' L ' + (-w) + ',' +   h  + ' Z';
        }

        case 'diamond': {
          var rx = r * 1.0, ry = r * 1.4;
          return 'M 0,' + (-ry) +
                 ' L '  + rx + ',0' +
                 ' L 0,' + ry +
                 ' L ' + (-rx) + ',0 Z';
        }

        case 'star': {
          var outerR = r * 1.3, innerR = r * 0.5, pts = [];
          for (var i = 0; i < 10; i++) {
            var angle = (Math.PI / 5) * i - Math.PI / 2;
            var rad   = (i % 2 === 0) ? outerR : innerR;
            pts.push(+(rad * Math.cos(angle)).toFixed(3) + ',' + +(rad * Math.sin(angle)).toFixed(3));
          }
          return 'M ' + pts.join(' L ') + ' Z';
        }

        case 'hexagon': {
          var pts = [];
          for (var i = 0; i < 6; i++) {
            var angle = (Math.PI / 3) * i - Math.PI / 6;
            pts.push(+(r * 1.1 * Math.cos(angle)).toFixed(3) + ',' + +(r * 1.1 * Math.sin(angle)).toFixed(3));
          }
          return 'M ' + pts.join(' L ') + ' Z';
        }

        case 'pentagon': {
          var pts = [];
          for (var i = 0; i < 5; i++) {
            var angle = (2 * Math.PI / 5) * i - Math.PI / 2;
            pts.push(+(r * 1.15 * Math.cos(angle)).toFixed(3) + ',' + +(r * 1.15 * Math.sin(angle)).toFixed(3));
          }
          return 'M ' + pts.join(' L ') + ' Z';
        }

        default:
          return null; // circle or ellipse — caller uses native SVG element
      }
    }

    /**
     * Appends the correct SVG shape child to every node <g>.
     * All shapes receive the CSS class "node-shape" for uniform selection.
     *   circle / ellipse  →  native SVG elements positioned via cx/cy in tick
     *   everything else   →  <path> positioned via transform="translate()" in tick
     */
    function appendShape(nodeSelection) {
      nodeSelection.each(function(d) {
        var g      = d3.select(this);
        var marker = (d.node_marker || 'circle').toLowerCase();
        var r      = parseFloat(d.node_size) || 8;
        var shape;

        if (marker === 'ellipse') {
          shape = g.append('ellipse')
            .attr('class', 'node-shape')
            .attr('rx', r * 1.6)
            .attr('ry', r);
        } else if (marker === 'circle' || shapePathD(marker, r) === null) {
          shape = g.append('circle')
            .attr('class', 'node-shape')
            .attr('r', r);
        } else {
          shape = g.append('path')
            .attr('class', 'node-shape')
            .attr('d', shapePathD(marker, r));
        }

        shape
          .style('fill',         d.node_color)
          .style('opacity',      d.node_opacity)
          .style('stroke-width', d.node_size_edge)
          .style('stroke',       d.node_color_edge);
      });
    }

    // ---- END SHAPE RENDERING ----

    //Append a SVG to the body of the html page. Assign this SVG as an object to svg
    var svg = d3.select("body").append("svg")
      .attr("width", width)
      .attr("height", height)
      .style("background-color", background_color)
      .call(d3.behavior.zoom().on("zoom", function () { svg.attr("transform", "translate(" + d3.event.translate + ")" + " scale(" + d3.event.scale + ")") }))
      .on("dblclick.zoom", null)
      .append("g")
    
    graphRec = JSON.parse(JSON.stringify(graph));
    
    //Creates the graph data structure out of the json data
    force.nodes(graph.nodes)
      .links(graph.links)
      .start();
    
    // Create all the line svgs but without locations yet
    var link = svg.selectAll(".link")
      .data(graph.links)
      .enter().append("line")
      .attr("class", "link")
      .attr('marker-start', function(d){ return 'url(#marker_' + d.marker_start + ')' })
      .attr("marker-end", function(d) {
        if (config.directed) {return 'url(#marker_' + d.marker_end + ')' }})
      .style("stroke-width", function(d) {return d.edge_width;})          // LINK-WIDTH
      .style("stroke", function(d) {return d.edge_color;})                 // EDGE-COLORS
      .style("stroke-dasharray", function(d) {return d.edge_style;})      // EDGE-STYLE
      .style("opacity", function(d) {return d.edge_opacity;})             // EDGE-OPACITY
    ;
    
    link.append("title").text(function(d) { return d.tooltip; });
    
    // ADD TEXT ON THE EDGES (PART 1/2)
    var linkText = svg.selectAll(".link-text")
      .data(graph.links)
      .enter().append("text")
      .attr("class", "link-text")
      .attr("font-size", function(d) {return d.label_fontsize + "px";})
      .style("fill", function(d) {return d.label_color;})
      .style("font-family", "Arial")
      .text(function(d) { return d.label; });
    
    //Do the same with the circles for the nodes
    var node = svg.selectAll(".node")
      .data(graph.nodes)
      .enter().append("g")
      .attr("class", "node")
      .call(drag)
      .on('dblclick', connectedNodes); // HIGHLIGHT ON/OFF
    
    // Right-click handler: release a pinned node back into the simulation (sticky mode only)
    if (sticky) {
      node.on('contextmenu', function(d) {
        d3.event.preventDefault();
        d.fixed = false;
        // Remove pinned visual cue
        d3.select(this).select(".node-shape")
          .style("stroke-dasharray", null)
          .style("stroke-width", function(d) { return d.node_size_edge; });
        force.resume();
      });
    }
    
    {{ CLICK_COMMENT }} node.on('click', color_on_click); // ON CLICK HANDLER

    // Append the correct shape per node (circle, ellipse, rect, triangle, etc.)
    appendShape(node);
    
    // Text in nodes
    node.append("text")
      .attr("dx",  function(d) { return config.node_text_inside ? 0 : 10; })
      .attr("dy", ".35em")
      .attr("text-anchor", function(d) { return config.node_text_inside ? "middle" : "start"; })
      .text(function(d) { return d.node_name; }) // NODE-TEXT
      .style("font-size", function(d) {
        if (config.node_text_inside) {
          // Auto-shrink font so the label fits within the circle diameter
          var r = parseFloat(d.node_size) || 8;
          var label = d.node_name || "";
          // Approximate: 0.6 * fontsize ≈ char width; fit label in 2*r with some padding
          var maxFontSize = parseFloat(d.node_fontsize) || 12;
          var fitSize = Math.min(maxFontSize, (1.8 * r) / Math.max(label.length * 0.6, 1));
          return Math.max(fitSize, 6) + "px";
        }
        return d.node_fontsize + "px"; // NODE FONT SIZE
      })
      .style("fill", function(d) {return d.node_fontcolor;}) // NODE FONT COLOR
      .style("font-family", "monospace")
      .style("pointer-events", "none");
    
    let showInHover = ["node_tooltip"]; // Tooltip
    node.append("title")
        .text((d) => Object.keys(d)
            .filter((key) => showInHover.indexOf(key) !== -1)
            .map((key) => `${d[key]}`)
            .join('\n')
        )
    
    //Now we are giving the SVGs co-ordinates - the force layout is generating the co-ordinates which this code is using to update the attributes of the SVG elements
    force.on("tick", function() {
      link.attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

      // Position each shape according to its SVG element type:
      //   circle / ellipse  →  cx / cy attributes
      //   path              →  transform translate(x, y)
      d3.selectAll(".node-shape").each(function(d) {
        var el  = d3.select(this);
        var tag = this.tagName.toLowerCase();
        if (tag === 'circle' || tag === 'ellipse') {
          el.attr("cx", d.x).attr("cy", d.y);
        } else {
          el.attr("transform", "translate(" + d.x + "," + d.y + ")");
        }
      });

      d3.selectAll("text").attr("x", function(d) { return d.x; })
        .attr("y", function(d) { return d.y; })
      linkText.attr("x", function(d) { return (d.source.x + d.target.x) / 2; })  // ADD TEXT ON THE EDGES (PART 2/2)
        .attr("y", function(d) { return (d.source.y + d.target.y) / 2; })
        .attr("text-anchor", "middle");
    
      node.each(collide(config.collision)); //COLLISION DETECTION. High means a big fight to get untouchable nodes (default=0.5)

  });

  // --------- MARKER FOR EDGE ENDINGS -----------

  var data_marker = [
    { id: 0, name: 'circle', path: 'M 0, 0  m -5, 0  a 5,5 0 1,0 10,0  a 5,5 0 1,0 -10,0', viewbox: '-6 -6 12 12' }
  , { id: 1, name: 'square', path: 'M 0,0 m -5,-5 L 5,-5 L 5,5 L -5,5 Z', viewbox: '-5 -5 10 10' }
  , { id: 2, name: 'arrow', path: 'M 0,0 m -5,-5 L 5,0 L -5,5 Z', viewbox: '-5 -5 10 10' }
  , { id: 3, name: 'stub', path: 'M 0,0 m -1,-5 L 1,-5 L 1,5 L -1,5 Z', viewbox: '-1 -5 2 10' }
  ]

  svg.append("defs").selectAll("marker")
    .data(data_marker)
    .enter()
    .append('svg:marker')
      .attr('id', function(d){ return 'marker_' + d.name})
      .attr('markerHeight', 10)
      .attr('markerWidth', 10)
      .attr("markerUnits", "userSpaceOnUse")                   // Fix marker width
      .attr('orient', 'auto')
      .attr('refX', 15)                                        // Offset marker-end
      .attr('refY', 0)
      .attr('viewBox', function(d){ return d.viewbox })
      .append('svg:path')
        .attr('d', function(d){ return d.path })               // Marker type
        .style("fill", '#808080')                              // Marker color
        .style("stroke", '#808080')                            // Marker edge-color
        .style("opacity", 0.95)                                // Marker opacity
        .style("stroke-width", 1);                             // Marker edge thickness

  // --------- END MARKER -----------


  // collision detection

  var padding = 1, // separation between circles
    radius = 8;

  function collide(alpha) {
    var quadtree = d3.geom.quadtree(graph.nodes);
    return function(d) {
      var rb = 2 * radius + padding,
        nx1 = d.x - rb,
        nx2 = d.x + rb,
        ny1 = d.y - rb,
        ny2 = d.y + rb;
      quadtree.visit(function(quad, x1, y1, x2, y2) {
        if (quad.point && (quad.point !== d)) {
          var x = d.x - quad.point.x,
            y = d.y - quad.point.y,
            l = Math.sqrt(x * x + y * y);
          if (l < rb) {
            l = (l - rb) / l * alpha;
            d.x -= x *= l;
            d.y -= y *= l;
            quad.point.x += x;
            quad.point.y += y;
          }
        }
        return x1 > nx2 || x2 < nx1 || y1 > ny2 || y2 < ny1;
      });
    };
  }
  // collision detection end


  //Toggle stores whether the highlighting is on
  var toggle = 0;
  //Create an array logging what is connected to what
  var linkedByIndex = {};
  for (i = 0; i < graph.nodes.length; i++) {
    linkedByIndex[i + "," + i] = 1;
  };
  graph.links.forEach(function(d) {
    linkedByIndex[d.source.index + "," + d.target.index] = 1;
  });
  //This function looks up whether a pair are neighbours
  function neighboring(a, b) {
    return linkedByIndex[a.index + "," + b.index];
  }


  // COLOR ON CLICK
  function color_on_click() {
    // Reset all nodes to their original styles
    d3.selectAll(".node")
    .select(".node-shape")
    .style("fill", function(d) {return d.node_color;})
    .style("opacity", function(d) {return d.node_opacity;})
    .style("stroke", function(d) {return d.node_color_edge;})
    .style("stroke-width", function(d) {return d.node_size_edge;})
    // Restore pinned cue on still-fixed nodes
    .style("stroke-dasharray", function(d) { return (sticky && d.fixed) ? "4,2" : null; });

    // Reset circle radii
    d3.selectAll(".node").select("circle.node-shape")
      .attr("r", function(d) { return d.node_size; });

    // Apply click styling to the selected node's shape
    var clickedShape = d3.select(this).select(".node-shape");
    clickedShape
      .style("fill", {{ CLICK_FILL }})
      .style("stroke", "{{ CLICK_STROKE }}")
      .style("stroke-width", {{ CLICK_STROKEW }});

    // Scale up the shape — strategy depends on element type
    var shapeNode = clickedShape.node();
    if (!shapeNode) return;
    var tag = shapeNode.tagName.toLowerCase();
    if (tag === 'circle') {
      clickedShape.attr("r", function(d) { return d.node_size * {{ CLICK_SIZE }}; });
    } else if (tag === 'ellipse') {
      clickedShape.each(function(d) {
        var r = parseFloat(d.node_size) || 8;
        d3.select(this)
          .attr("rx", r * 1.6 * {{ CLICK_SIZE }})
          .attr("ry", r * {{ CLICK_SIZE }});
      });
    } else {
      // <path>: re-compute path with a scaled radius
      clickedShape.each(function(d) {
        var r = (parseFloat(d.node_size) || 8) * {{ CLICK_SIZE }};
        d3.select(this).attr("d", shapePathD(d.node_marker, r));
      });
    }
  }



  function connectedNodes() {
    if (toggle == 0) {
      //Reduce the opacity of all but the neighbouring nodes
      d = d3.select(this).node().__data__;
      node.style("opacity", function(o) {
        return neighboring(d, o) | neighboring(o, d) ? 1 : 0.1;
      });
      link.style("opacity", function(o) {
        return d.index == o.source.index | d.index == o.target.index ? 1 : 0.1;
      });
      toggle = 1;
    } else {
      //Put them back to opacity=1
      node.style("opacity", 0.95);
      link.style("opacity", 1);

      toggle = 0;
    }
  }


  //adjust threshold
  function threshold() {
    let thresh = this.value;

    graph.links.splice(0, graph.links.length);
    linkText = linkText.data([]);   // CLEAR EDGE-LABELS: Clear the linkText selection
    linkText.exit().remove();       // CLEAR EDGE-LABELS: Clear the linkText elements from the DOM

    for (var i = 0; i < graphRec.links.length; i++) {
      if (graphRec.links[i].edge_weight > thresh) {
        graph.links.push(graphRec.links[i]);
      }
    }
    restart();
  }

  // Set the initial value of the slider to the user-defined threshold
  document.getElementById('thresholdSlider').value = {{ SET_SLIDER }};
  // Call the threshold function to set the network state
  threshold.call(document.getElementById('thresholdSlider'));

  d3.select("#thresholdSlider").on("change", threshold);

  //Restart the visualisation after any node and link changes
  function restart() {

    // Update EDGE-LINKS
    link = link.data(graph.links);
    link.exit().remove();
    link.enter().insert("line", ".node").attr("class", "link");
    link.style("stroke-width", function(d) {return d.edge_width;});           // LINK-WIDTH AFTER BREAKING WITH SLIDER
    link.style("marker-end", function(d) {                                    // Include the markers.
      if (config.directed) {return 'url(#marker_' + d.marker_end + ')' }})
    link.style("stroke", function(d) {return d.edge_color;});                 // EDGE-COLOR AFTER BREAKING WITH SLIDER
    link.style("stroke-dasharray", function(d) {return d.edge_style;})        // EDGE-STYLE
    link.style("opacity", function(d) {return d.edge_opacity;});              // EDGE-OPACITY AFTER BREAKING WITH SLIDER

    // Update EDGE-LABELS
    linkText = linkText.data(graph.links);
    linkText.exit().remove();
    linkText.enter().append("text")
      .attr("class", "link-text")
      .attr("font-size", function(d) {return d.label_fontsize + "px";})
      .style("fill", function(d) {return d.label_color;})
      .style("font-family", "Arial")
      .text(function(d) { return d.label; });

    node = node.data(graph.nodes);
    node.enter().insert("circle", ".cursor").attr("class", "node").attr("r", 5).call(force.drag);
    force.start();
  }

}